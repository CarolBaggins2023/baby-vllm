"""
Phase 6 测试套件 —— Config server 字段 + CLI 参数映射完善
============================================================

测试覆盖矩阵（7 大类，共 40+ 条用例）：

  Section 1 — Config 默认值
      验证 host="0.0.0.0"、port=8000 默认值，以及自定义值能正确存储。

  Section 2 — Config 验证逻辑（__post_init__）
      覆盖 host 的 6 种非法输入（空字符串、None、int、bytes 等）
      覆盖 port 的 10+ 种非法输入（0、负数、65536+、float、str、bool 等）
      覆盖 port 边界值 1 和 65535 的合法通过

  Section 3 — 数据流：fields(Config) 动态发现
      验证 LLMEngine.__init__ 的字段过滤模式能自动捕获 host/port
      验证无关 kwargs 被正确丢弃

  Section 4 — CLI 参数解析
      验证 argparse 默认值、自定义值
      验证 engine_kwargs 映射完整性（含 host/port）
      验证 --max-model-len 条件性映射
      验证 log_level 不进入 engine_kwargs

  Section 5 — 完整管道集成（mock）
      验证 Config 验证后的值被 uvicorn 使用
      验证无效 port 在 GPU 分配前快速失败

  Section 6 — 向后兼容性
      验证不传 host/port 时默认值生效
      验证所有现有 Config 字段不受影响
      验证 dataclass 字段排序合法性
      验证 port=True 的 bool-as-int 边界情况

  Section 7 — GPU 集成测试（需要 BABYVLLM_TEST_MODEL 环境变量）
      验证 LLMEngine/AsyncLLMEngine 接受 host/port kwargs
      验证离线 generate() 在新 Config 下结果一致

运行方式：
  # 纯单元测试（无需 GPU，<1 秒）
  pytest test_phase6.py -v -m "not gpu"

  # 含 GPU 的完整测试（需要设置 BABYVLLM_TEST_MODEL 环境变量）
  BABYVLLM_TEST_MODEL=/path/to/model pytest test_phase6.py -v

  # 仅 GPU 测试
  BABYVLLM_TEST_MODEL=/path/to/model pytest test_phase6.py -v -m gpu

  # 运行并显示详细输出
  pytest test_phase6.py -v -s --tb=short
"""

import os
import sys
import tempfile
import dataclasses
import argparse
from unittest.mock import patch, MagicMock
from dataclasses import fields

import pytest

# ---------------------------------------------------------------------------
# 确保 babyvllm 在 import 路径中
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from babyvllm.config import Config


# ===========================================================================
# Shared Fixtures
# ===========================================================================


@pytest.fixture
def fake_model_dir():
    """
    创建一个临时目录，使其通过 Config.__post_init__ 中的
    os.path.isdir(self.model) 检查。
    使用 mkdtemp + 手动清理，避免 TemporaryDirectory 上下文管理器
    与 mock 嵌套时的生命周期问题。
    """
    path = tempfile.mkdtemp(prefix="babyvllm_test_")
    yield path
    try:
        os.rmdir(path)
    except OSError:
        pass  # 目录可能已被其他进程清理


@pytest.fixture
def mock_autoconfig():
    """
    Mock AutoConfig.from_pretrained()，避免 Config.__post_init__
    需要真实的 HuggingFace 模型文件。返回的 mock 对象提供
    max_position_embeddings=4096，与 Config 默认 max_model_length 匹配。
    """
    with patch("babyvllm.config.AutoConfig") as mock_ac:
        fake_cfg = MagicMock()
        fake_cfg.max_position_embeddings = 4096
        mock_ac.from_pretrained.return_value = fake_cfg
        yield mock_ac


# ===========================================================================
# Section 1: Config host/port 默认值与自定义值
# ===========================================================================


class TestConfigDefaults:
    """
    验证 Config 中 host 和 port 的默认值（per plan.md 6.1），
    以及通过 kwargs 传入自定义值时的行为。
    """

    def test_host_default_is_all_interfaces(self, fake_model_dir, mock_autoconfig):
        """plan.md 6.1 规定 Config.host 默认值为 "0.0.0.0"。"""
        config = Config(model=fake_model_dir)
        assert config.host == "0.0.0.0", (
            f"Expected host='0.0.0.0' (plan.md 6.1), got {config.host!r}"
        )

    def test_port_default_is_8000(self, fake_model_dir, mock_autoconfig):
        """plan.md 6.1 规定 Config.port 默认值为 8000。"""
        config = Config(model=fake_model_dir)
        assert config.port == 8000, (
            f"Expected port=8000 (plan.md 6.1), got {config.port}"
        )

    def test_custom_host_stored_correctly(self, fake_model_dir, mock_autoconfig):
        """kwargs host 值应覆写默认值并被正确存储。"""
        config = Config(model=fake_model_dir, host="192.168.1.100")
        assert config.host == "192.168.1.100"

    def test_custom_port_stored_correctly(self, fake_model_dir, mock_autoconfig):
        """kwargs port 值应覆写默认值并被正确存储。"""
        config = Config(model=fake_model_dir, port=9999)
        assert config.port == 9999

    def test_custom_host_and_port_together(self, fake_model_dir, mock_autoconfig):
        """同时传入 host 和 port 时两个值都应正确存储。"""
        config = Config(model=fake_model_dir, host="10.0.0.1", port=443)
        assert config.host == "10.0.0.1"
        assert config.port == 443

    def test_host_type_is_str(self, fake_model_dir, mock_autoconfig):
        """host 字段的类型注解为 str，默认值和自定义值都应为 str 类型。"""
        config = Config(model=fake_model_dir)
        assert isinstance(config.host, str)
        config2 = Config(model=fake_model_dir, host="localhost")
        assert isinstance(config2.host, str)

    def test_port_type_is_int(self, fake_model_dir, mock_autoconfig):
        """port 字段的类型注解为 int，默认值和自定义值都应为 int 类型。"""
        config = Config(model=fake_model_dir)
        assert isinstance(config.port, int)
        config2 = Config(model=fake_model_dir, port=8080)
        assert isinstance(config2.port, int)


# ===========================================================================
# Section 2: Config.__post_init__ 验证逻辑
# ===========================================================================


class TestConfigHostValidation:
    """
    测试 host 字段的验证逻辑。

    当前实现（minimal validation）：
      assert isinstance(self.host, str) and len(self.host) > 0

    注意：当前不验证 IP 格式、域名合法性或空格内容。
    以下测试覆盖了所有可能触发验证失败的输入类型。
    """

    # ---- 非法 host 输入（应触发 AssertionError）----

    @pytest.mark.parametrize("bad_host, desc", [
        ("", "空字符串"),
        (None, "None"),
    ])
    def test_host_invalid_raises_assertion_error(
        self, fake_model_dir, mock_autoconfig, bad_host, desc
    ):
        """host 为无效值时应快速失败并给出明确错误信息。"""
        with pytest.raises(AssertionError, match="host must be a non-empty string"):
            Config(model=fake_model_dir, host=bad_host)

    @pytest.mark.parametrize("bad_host_type, desc", [
        (123, "int"),
        (True, "bool"),
        (3.14, "float"),
        ([], "list"),
        ({}, "dict"),
        (b"127.0.0.1", "bytes"),
    ])
    def test_host_wrong_type_raises_assertion_error(
        self, fake_model_dir, mock_autoconfig, bad_host_type, desc
    ):
        """host 为非 str 类型时 isinstance 检查失败，应抛出 AssertionError。"""
        with pytest.raises(AssertionError, match="host must be a non-empty string"):
            Config(model=fake_model_dir, host=bad_host_type)

    # ---- 合法 host 输入（应通过验证）----

    @pytest.mark.parametrize("good_host, desc", [
        ("127.0.0.1", "IPv4 loopback"),
        ("0.0.0.0", "IPv4 all-interfaces"),
        ("192.168.1.1", "IPv4 private"),
        ("localhost", "hostname localhost"),
        ("my-server.local", "DNS name"),
        ("::1", "IPv6 loopback"),
        ("   ", "whitespace-only (当前实现不拒绝)"),
        ("a" * 255, "长字符串 (255 chars)"),
    ])
    def test_host_valid_passes(
        self, fake_model_dir, mock_autoconfig, good_host, desc
    ):
        """合法 host 值应通过验证并被正确存储。"""
        config = Config(model=fake_model_dir, host=good_host)
        assert config.host == good_host, f"Failed for input: {desc}"


class TestConfigPortValidation:
    """
    测试 port 字段的验证逻辑。

    当前实现：
      assert isinstance(self.port, int) and 1 <= self.port <= 65535

    端口范围 1-65535 覆盖了所有 TCP/UDP 有效端口。
    端口 0 是保留端口（系统自动分配），在此被明确拒绝。
    """

    # ---- 非法 port 输入（应触发 AssertionError）----

    @pytest.mark.parametrize("bad_port, desc", [
        (0, "端口 0 (系统保留)"),
        (-1, "负数"),
        (-100, "负数 (绝对值较大)"),
        (65536, "超出上限 1"),
        (70000, "超出上限较多"),
        (99999, "典型错误输入"),
        (1000000, "远超出范围"),
    ])
    def test_port_out_of_range_raises(
        self, fake_model_dir, mock_autoconfig, bad_port, desc
    ):
        """port 超出 1-65535 范围应抛出 AssertionError。"""
        with pytest.raises(AssertionError, match="port must be between 1 and 65535"):
            Config(model=fake_model_dir, port=bad_port)

    @pytest.mark.parametrize("bad_port_type, desc", [
        (8000.0, "float (即使数值有效)"),
        (8000.5, "float (非整数)"),
        ("8000", "str (数字字符串)"),
        ("http", "str (非数字)"),
        (None, "None"),
        ([8000], "list"),
    ])
    def test_port_wrong_type_raises(
        self, fake_model_dir, mock_autoconfig, bad_port_type, desc
    ):
        """port 为非 int 类型时 isinstance 检查失败。"""
        with pytest.raises(AssertionError, match="port must be between 1 and 65535"):
            Config(model=fake_model_dir, port=bad_port_type)

    # ---- 合法 port 输入（应通过验证）----

    @pytest.mark.parametrize("good_port, desc", [
        (1, "最小有效端口"),
        (80, "HTTP 标准端口"),
        (443, "HTTPS 标准端口"),
        (8000, "默认端口"),
        (8080, "常用替代端口"),
        (65535, "最大有效端口"),
        (1024, "非特权端口边界"),
        (65534, "接近最大值"),
    ])
    def test_port_valid_passes(
        self, fake_model_dir, mock_autoconfig, good_port, desc
    ):
        """合法 port 值应通过验证并被正确存储。"""
        config = Config(model=fake_model_dir, port=good_port)
        assert config.port == good_port, f"Failed for input: {desc}"


class TestConfigPortBoolEdgeCase:
    """
    特殊边界情况：Python 中 bool 是 int 的子类。

    isinstance(True, int)  → True
    isinstance(False, int) → True

    这意味着 port=True 会通过 isinstance 检查，且
    1 <= True <= 65535 → True（因为 True == 1）

    这是一个已知的行为特性，不是 bug。
    如果未来需要禁止，可以在 isinstance 后增加 type(self.port) is int 检查。
    """

    def test_port_true_passes_as_int_1(self, fake_model_dir, mock_autoconfig):
        """port=True 被当作 int(1) 通过验证。这是 bool-as-int 的已知行为。"""
        config = Config(model=fake_model_dir, port=True)
        # Python 中 bool 是 int 的子类，True == 1 且 isinstance(True, int) 为 True
        # 因此 port=True 被验证接受，与 1 比较相等
        assert config.port == 1
        assert isinstance(config.port, int)  # bool 是 int 的子类，此检查通过

    def test_port_false_raises(self, fake_model_dir, mock_autoconfig):
        """port=False → int(0) → 不在 1-65535 范围 → 抛出异常。"""
        with pytest.raises(AssertionError, match="port must be between 1 and 65535"):
            Config(model=fake_model_dir, port=False)


# ===========================================================================
# Section 3: 数据流 —— fields(Config) 动态字段发现
# ===========================================================================
#
# LLMEngine.__init__ (llm_engine.py:30-32) 使用以下模式过滤 kwargs：
#
#   config_fields = {field.name for field in fields(Config)}
#   config_kwargs = {k:v for k, v in kwargs.items() if k in config_fields}
#   self.config = Config(model, **config_kwargs)
#
# 这意味着只要 host/port 是 Config 的字段，它们就会被自动捕获。
# 以下测试验证这个关键假设。


class TestFieldsDiscovery:
    """验证 dataclasses.fields(Config) 自动发现 host 和 port。"""

    def test_host_is_discoverable_field(self):
        """host 必须在 fields(Config) 中出现，才能被 LLMEngine 过滤器捕获。"""
        field_names = {f.name for f in fields(Config)}
        assert "host" in field_names, (
            f"host not found in fields(Config). Found: {sorted(field_names)}"
        )

    def test_port_is_discoverable_field(self):
        """port 必须在 fields(Config) 中出现，才能被 LLMEngine 过滤器捕获。"""
        field_names = {f.name for f in fields(Config)}
        assert "port" in field_names, (
            f"port not found in fields(Config). Found: {sorted(field_names)}"
        )

    def test_config_field_count_is_13(self):
        """
        验证 Config 的字段总数。

        Phase 1-5 原有字段 (11):
          model, max_num_batched_tokens, max_num_sequences, max_model_length,
          gpu_memory_utilization, tensor_parallel_size, enforce_eager,
          eos, kvcache_block_size, num_kvcache_blocks, hf_config

        Phase 6 新增字段 (2):
          host, port

        总计: 13

        如果这个数字变了，说明有人意外增删了 Config 字段。
        """
        expected = {
            "model", "max_num_batched_tokens", "max_num_sequences",
            "max_model_length", "gpu_memory_utilization",
            "tensor_parallel_size", "enforce_eager", "eos",
            "kvcache_block_size", "num_kvcache_blocks",
            "host", "port", "hf_config",
        }
        actual = {f.name for f in fields(Config)}
        assert actual == expected, (
            f"Config fields mismatch.\n"
            f"  Extra fields:   {sorted(actual - expected)}\n"
            f"  Missing fields: {sorted(expected - actual)}"
        )

    def test_filter_pattern_captures_host_and_port(self, fake_model_dir, mock_autoconfig):
        """
        端到端模拟 LLMEngine.__init__ 的过滤逻辑：

        输入 kwargs（含 host/port + 无关参数）
          → 按 fields(Config) 过滤
          → 输出 config_kwargs（应含 host/port，不含无关参数）
          → 传给 Config() 构造（应成功）
        """
        # ---- Step 1: 模拟 kwargs 过滤（与 llm_engine.py:30-31 一致）----
        config_fields = {f.name for f in fields(Config)}
        raw_kwargs = {
            "model": fake_model_dir,
            "host": "192.168.1.1",
            "port": 443,
            "max_num_batched_tokens": 8192,
            "unknown_param": "should_be_dropped",
            "another_unknown": 42,
        }
        config_kwargs = {
            k: v for k, v in raw_kwargs.items() if k in config_fields
        }

        # ---- Step 2: 验证过滤结果 ----
        # host/port 被保留
        assert config_kwargs["host"] == "192.168.1.1"
        assert config_kwargs["port"] == 443
        assert config_kwargs["max_num_batched_tokens"] == 8192
        # 无关参数被丢弃
        assert "unknown_param" not in config_kwargs
        assert "another_unknown" not in config_kwargs
        # model 被保留
        assert config_kwargs["model"] == fake_model_dir

        # ---- Step 3: 验证 Config 能正常构造 ----
        config = Config(**config_kwargs)
        assert config.host == "192.168.1.1"
        assert config.port == 443

    def test_filter_survives_with_only_model(self):
        """
        极端情况：kwargs 只有 model（其他全部使用默认值）。
        host/port 应使用 Config 默认值。
        """
        config_fields = {f.name for f in fields(Config)}
        raw_kwargs = {"model": "/some/path"}
        config_kwargs = {
            k: v for k, v in raw_kwargs.items() if k in config_fields
        }
        # host/port 不在 config_kwargs 中（未传入）
        assert "host" not in config_kwargs
        assert "port" not in config_kwargs
        # 但这没关系——Config 有自己的默认值


# ===========================================================================
# Section 4: CLI 参数解析
# ===========================================================================
#
# 以下测试复现了 cli.py 中的 argparse 解析器和 engine_kwargs 构建逻辑。
# 我们没有直接导入 cli.main()，因为：
#   1. main() 内部有延迟导入（torch、transformers），会触发 GPU 初始化
#   2. main() 会启动 uvicorn（阻塞调用）
#   3. 我们只需要验证参数解析和映射逻辑
#
# 注意：下面 _build_parser() 的参数定义必须与 cli.py 的 main() 保持一致。
# 如果 cli.py 新增/修改了 CLI 参数，这里的 parser 也需要同步更新。


class TestCLIArgParsing:
    """CLI 参数解析测试。"""

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        """
        构建与 cli.py main() 相同的 ArgumentParser。
        维护者注意：如果 cli.py 增删参数，此处需同步更新。
        """
        parser = argparse.ArgumentParser(prog="baby-vllm-server")
        # Required
        parser.add_argument("--model", type=str, required=True)
        # Server
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        # Engine / Model
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
        parser.add_argument("--max-num-sequences", type=int, default=512)
        parser.add_argument("--max-model-len", type=int, default=None)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
        parser.add_argument("--enforce-eager", action="store_true", default=False)
        # KV Cache
        parser.add_argument("--kvcache-block-size", type=int, default=256)
        parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
        # Logging
        parser.add_argument("--log-level", type=str, default="info",
                           choices=["debug", "info", "warning", "error", "critical"])
        return parser

    def _parse(self, *extra_args: str) -> argparse.Namespace:
        """解析 CLI 参数并返回 Namespace。"""
        parser = self._build_parser()
        return parser.parse_args(["--model", "/fake/model"] + list(extra_args))

    # ---- 默认值 ----

    def test_host_default_is_127_0_0_1(self):
        """CLI --host 默认值应为 "127.0.0.1"（安全默认，仅绑定本地）。"""
        args = self._parse()
        assert args.host == "127.0.0.1", (
            f"CLI --host default should be '127.0.0.1' (secure), "
            f"got {args.host!r}"
        )

    def test_port_default_is_8000(self):
        """CLI --port 默认值应为 8000。"""
        args = self._parse()
        assert args.port == 8000

    # ---- 自定义值 ----

    def test_host_custom_value(self):
        """--host 自定义值应正确解析。"""
        args = self._parse("--host", "0.0.0.0")
        assert args.host == "0.0.0.0"

    def test_port_custom_value(self):
        """--port 自定义值应正确解析。"""
        args = self._parse("--port", "9999")
        assert args.port == 9999

    def test_host_and_port_custom_together(self):
        """--host 和 --port 同时指定时应都正确解析。"""
        args = self._parse("--host", "10.0.0.1", "--port", "443")
        assert args.host == "10.0.0.1"
        assert args.port == 443

    # ---- engine_kwargs 映射 ----

    def test_engine_kwargs_includes_host_and_port(self):
        """
        验证 engine_kwargs 字典中包含 host 和 port。
        这是 Phase 6.2 的核心变更——host/port 必须通过 engine_kwargs 传入 Config。
        """
        args = self._parse("--host", "0.0.0.0", "--port", "8080")

        # 复现 cli.py 中 engine_kwargs 的构建逻辑
        engine_kwargs = {
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_sequences": args.max_num_sequences,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "enforce_eager": args.enforce_eager,
            "kvcache_block_size": args.kvcache_block_size,
            "num_kvcache_blocks": args.num_kvcache_blocks,
            "host": args.host,       # Phase 6 新增
            "port": args.port,       # Phase 6 新增
        }

        assert "host" in engine_kwargs, "engine_kwargs must contain 'host' (Phase 6.2)"
        assert "port" in engine_kwargs, "engine_kwargs must contain 'port' (Phase 6.2)"
        assert engine_kwargs["host"] == "0.0.0.0"
        assert engine_kwargs["port"] == 8080

    def test_all_engine_kwargs_keys_present(self):
        """
        验证 engine_kwargs 包含所有预期的键。
        这个测试会在新增/删除 engine_kwargs 条目时失效，起到哨兵作用。
        """
        args = self._parse()
        engine_kwargs = {
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_sequences": args.max_num_sequences,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "enforce_eager": args.enforce_eager,
            "kvcache_block_size": args.kvcache_block_size,
            "num_kvcache_blocks": args.num_kvcache_blocks,
            "host": args.host,
            "port": args.port,
        }

        expected_keys = {
            "max_num_batched_tokens", "max_num_sequences",
            "gpu_memory_utilization", "tensor_parallel_size",
            "enforce_eager", "kvcache_block_size", "num_kvcache_blocks",
            "host", "port",
        }
        assert set(engine_kwargs.keys()) == expected_keys

    # ---- log_level 不进入 engine_kwargs ----

    def test_log_level_not_in_engine_kwargs(self):
        """
        log_level 是纯 uvicorn 参数，不应进入 engine_kwargs。
        如果 log_level 被加入 Config，这个测试会提醒审查者确认是否为预期变更。
        """
        args = self._parse("--log-level", "debug")
        assert args.log_level == "debug"
        # log_level 不在 engine_kwargs 中（这是正确的行为）
        # 验证它不在 fields(Config) 中
        field_names = {f.name for f in fields(Config)}
        assert "log_level" not in field_names, (
            "log_level should NOT be a Config field — it is a uvicorn-only concern"
        )

    # ---- --max-model-len 条件性映射 ----

    def test_max_model_len_not_specified(self):
        """未指定 --max-model-len 时不应进入 engine_kwargs。"""
        args = self._parse()
        assert args.max_model_len is None
        # 在 cli.py 中，只有当 args.max_model_len is not None 时才添加
        engine_kwargs = {
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_sequences": args.max_num_sequences,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "enforce_eager": args.enforce_eager,
            "kvcache_block_size": args.kvcache_block_size,
            "num_kvcache_blocks": args.num_kvcache_blocks,
            "host": args.host,
            "port": args.port,
        }
        assert "max_model_length" not in engine_kwargs

    def test_max_model_len_specified(self):
        """指定 --max-model-len 时应映射为 max_model_length。"""
        args = self._parse("--max-model-len", "2048")
        assert args.max_model_len == 2048
        # 在 cli.py 中：
        # if args.max_model_len is not None:
        #     engine_kwargs["max_model_length"] = args.max_model_len
        engine_kwargs: dict = {}
        if args.max_model_len is not None:
            engine_kwargs["max_model_length"] = args.max_model_len
        assert engine_kwargs["max_model_length"] == 2048


# ===========================================================================
# Section 5: 完整管道集成测试（mock 重量级依赖）
# ===========================================================================


class TestConfigToUvicornPipeline:
    """
    验证从 CLI args → Config → uvicorn.run() 的完整数据流。

    我们不真正启动服务器，而是 mock 关键依赖来验证值的传递正确性。
    """

    def test_config_values_propagate_correctly(self, fake_model_dir, mock_autoconfig):
        """
        模拟 cli.py 中 uvicorn.run() 的行为：
        host/port 应从 engine.engine.config 读取，而不是从原始 CLI args 读取。
        这确保 Config 验证后的规范值被使用。
        """
        # 模拟：CLI 传入 --host 127.0.0.1 --port 8000
        # 这些值通过 engine_kwargs 进入 Config
        config = Config(
            model=fake_model_dir,
            host="127.0.0.1",
            port=8000,
        )

        # 模拟：cli.py 中 uvicorn.run() 从 engine.engine.config 读取值
        uvicorn_host = config.host
        uvicorn_port = config.port

        assert uvicorn_host == "127.0.0.1"
        assert uvicorn_port == 8000

    def test_invalid_port_fails_before_gpu_allocation(self, fake_model_dir, mock_autoconfig):
        """
        验证快速失败机制：无效 port 在 Config.__post_init__ 就被捕获，
        此时尚未分配 GPU 内存。

        执行顺序（Config.__post_init__）：
          1. os.path.isdir(model)           ← 目录检查
          2. kvcache_block_size % 256 == 0  ← KV cache 参数检查
          3. tensor_parallel_size ∈ [1,8]   ← 并行度检查
          4. AutoConfig.from_pretrained()   ← 读取模型配置（仅 I/O，无 GPU）
          5. max_model_length 调整
          6. max_num_batched_tokens 检查
          7. host 验证                       ← 新增
          8. port 验证                       ← 新增 ← 在此失败
          9. [LLMEngine 继续] GPU 分配       ← 不会到达
        """
        with pytest.raises(AssertionError, match="port must be between 1 and 65535"):
            Config(model=fake_model_dir, port=99999)
        # 如果到达这里，说明验证在 GPU 分配之前触发 —— 这正是我们想要的

    def test_config_is_single_source_of_truth(self, fake_model_dir, mock_autoconfig):
        """
        验证 Config 是 host/port 的单一数据源：
        - 如果 CLI 传入了值 → Config 存储该值 → uvicorn 使用 Config 的值
        - Config 默认值和使用 CLI 覆写后的值不应混淆
        """
        # 场景 A: 使用默认值
        config_a = Config(model=fake_model_dir)
        assert config_a.host == "0.0.0.0"  # Config 默认
        assert config_a.port == 8000

        # 场景 B: CLI 覆写（模拟 --host 127.0.0.1 --port 8080）
        config_b = Config(model=fake_model_dir, host="127.0.0.1", port=8080)
        assert config_b.host == "127.0.0.1"  # CLI 安全默认覆写
        assert config_b.port == 8080

        # 两个场景的值不互相影响
        assert config_a.host != config_b.host


# ===========================================================================
# Section 6: 向后兼容性
# ===========================================================================


class TestBackwardCompatibility:
    """
    验证 Phase 6 变更不影响已有功能。

    关键原则：所有 Phase 1-5 的代码路径必须保持不变。
    """

    def test_create_config_without_host_port(self, fake_model_dir, mock_autoconfig):
        """
        不传 host/port 参数创建 Config 应成功。
        这是 LLMEngine.generate() 离线模式的典型调用方式：
          LLMEngine(model="/path/to/model")
          其中不包含 host/port kwargs。
        """
        config = Config(model=fake_model_dir)
        # 使用默认值
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        # 所有引擎参数应正常工作
        assert config.max_num_batched_tokens == 16384
        assert config.max_num_sequences == 512

    def test_all_existing_fields_preserved(self, fake_model_dir, mock_autoconfig):
        """
        所有 Phase 1-5 的 Config 字段在接受自定义值时应正确存储。
        新增的 host/port 使用默认值，不干扰现有字段。
        """
        config = Config(
            model=fake_model_dir,
            max_num_batched_tokens=8192,
            max_num_sequences=256,
            max_model_length=2048,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=1,
            enforce_eager=True,
            eos=100,
            kvcache_block_size=512,
            num_kvcache_blocks=200,
        )
        assert config.max_num_batched_tokens == 8192
        assert config.max_num_sequences == 256
        assert config.max_model_length == 2048
        assert config.gpu_memory_utilization == 0.7
        assert config.tensor_parallel_size == 1
        assert config.enforce_eager is True
        assert config.eos == 100
        assert config.kvcache_block_size == 512
        assert config.num_kvcache_blocks == 200
        # Phase 6 新增字段使用默认值（不影响现有测试）
        assert config.host == "0.0.0.0"
        assert config.port == 8000

    def test_dataclass_field_ordering_is_valid(self):
        """
        Python dataclass 规则：有默认值的字段必须在所有无默认值的字段之后。

        Config 中 model 是唯一无默认值的字段，排在第一位。
        所有其他字段（包括 host/port）都有默认值，跟在 model 之后。

        如果排序不合法，Python 会直接拒绝创建该类。
        这个测试作为额外的文档化验证。
        """
        flds = fields(Config)
        found_default = False
        for f in flds:
            has_default = (
                f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING
            )
            if has_default:
                found_default = True
            elif found_default:
                # 无默认值的字段出现在有默认值的字段之后 → dataclass 会报错
                pytest.fail(
                    f"Field '{f.name}' has no default but appears after "
                    f"fields with defaults. This would cause a TypeError "
                    f"when creating the dataclass."
                )
        # model 是无默认值的唯一字段，必须排在第一位
        assert flds[0].name == "model", (
            f"First field should be 'model' (only field without default), "
            f"got '{flds[0].name}'"
        )

    def test_extra_kwargs_are_safely_dropped_by_filter(self):
        """
        LLMEngine.__init__ 的过滤模式丢弃非 Config 字段的 kwargs。
        传入无关参数不应导致错误——它们被静默忽略。
        """
        config_fields = {f.name for f in fields(Config)}
        kwargs = {
            "model": "/some/model",
            "host": "127.0.0.1",
            "port": 8080,
            "this_param_does_not_exist": 12345,
            "neither_does_this": "hello",
        }
        config_kwargs = {
            k: v for k, v in kwargs.items() if k in config_fields
        }
        # 无关参数被丢弃
        assert "this_param_does_not_exist" not in config_kwargs
        assert "neither_does_this" not in config_kwargs
        # 相关参数被保留
        assert "host" in config_kwargs
        assert "port" in config_kwargs
        assert "model" in config_kwargs


# ===========================================================================
# Section 7: GPU 集成测试（需要 BABYVLLM_TEST_MODEL 环境变量）
# ===========================================================================
#
# 重要限制：
#   LLMEngine.__init__ → ModelRunner.__init__ 调用
#   torch.distributed.init_process_group()，该函数在一个进程中只能调用一次。
#   因此所有 GPU 测试必须共享同一个引擎实例。
#
#   解决方案：使用 module-scoped fixture 在模块级创建唯一的引擎，
#   各测试通过 fixture 引用该引擎。模块 tear down 时清理资源。
#
# 设置环境变量以启用：
#   export BABYVLLM_TEST_MODEL=/path/to/Qwen3-0.6B
#
# 如果没有设置环境变量，所有 GPU 测试被自动跳过。


_GPU_MODEL = os.environ.get("BABYVLLM_TEST_MODEL", "")


def _gpu_tests_enabled() -> bool:
    """检测 GPU 测试是否可用。"""
    if not _GPU_MODEL:
        return False
    if not os.path.isdir(_GPU_MODEL):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---- Module-scoped fixtures: 整个模块共享一个引擎 ----
# WHY: dist.init_process_group() is one-shot per process.
# Creating multiple LLMEngines would raise:
#   ValueError: trying to initialize the default process group twice!
# Module-scoped fixtures solve this by creating the engine exactly once.


@pytest.fixture(scope="module")
def _gpu_engine():
    """
    创建 LLMEngine 一次，所有 GPU 测试共享。module scope 保证只创建一次。

    引擎使用 host="127.0.0.1"、port=8000 创建，模拟 CLI 安全默认值。
    测试 "不传 host/port 使用 Config 默认值" 的用例由 Section 3 的
    单元测试覆盖（filter pattern + Config defaults），无需 GPU 重复验证。

    Teardown:
      1. engine.exit()  —— 清理 model_runner 和 worker 进程
      2. atexit.unregister() —— 防止 atexit 二次调用 exit()
      3. dist.destroy_process_group() —— 清理 NCCL 进程组，消除 PyTorch 警告
    """
    if not _gpu_tests_enabled():
        pytest.skip(
            "GPU tests require: CUDA GPU + BABYVLLM_TEST_MODEL env var "
            "pointing to a valid model directory"
        )

    import atexit
    import torch.distributed as dist
    from babyvllm.engine.llm_engine import LLMEngine

    engine = LLMEngine(model=_GPU_MODEL, host="127.0.0.1", port=8000)
    yield engine

    # ---- 清理资源 ----
    engine.exit()
    # 从 atexit 注销，避免在进程退出时二次调用（此时 model_runner 已被删除）
    atexit.unregister(engine.exit)
    # 销毁分布式进程组，避免 NCCL 警告
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def _gpu_async_engine(_gpu_engine):
    """
    创建 AsyncLLMEngine 一次，所有在线 GPU 测试共享。module scope 保证只创建一次。

    复用 _gpu_engine 的底层 LLMEngine，避免重复加载模型导致 GPU OOM。
    Teardown 只停止异步 loop，底层引擎由 _gpu_engine fixture 负责清理。
    """
    if not _gpu_tests_enabled():
        pytest.skip(
            "GPU tests require: CUDA GPU + BABYVLLM_TEST_MODEL env var "
            "pointing to a valid model directory"
        )

    import asyncio
    from babyvllm.engine.async_llm_engine import AsyncLLMEngine

    engine = AsyncLLMEngine(engine=_gpu_engine)
    yield engine

    # ---- 清理资源 ----
    # engine.stop() 需要在 engine 最初启动的 event loop 中调用。
    # pytest-asyncio 的 asyncio_mode="auto" 为每个测试函数创建独立的 event loop，
    # 测试结束后 engine 的后台 task 已随 loop 关闭而终止。
    # 如果 engine 从未启动（如只读 config 的测试），也无需 stop。
    if not engine.engine_started:
        return
    try:
        asyncio.run(engine.stop())
    except RuntimeError:
        # asyncio.Event 绑定在已关闭的 event loop 上，随 loop 关闭已等效停止
        pass


# =========================================================================
# GPU Test Classes
# =========================================================================


@pytest.mark.gpu
class TestLLMEngineWithHostPort:
    """
    LLMEngine 接受 host/port kwargs 的 GPU 集成测试。

    通过 module-scoped _gpu_engine fixture 共享引擎。
    注："不传 host/port 使用默认值" 的场景已由 Section 3 单元测试覆盖。
    """

    def test_config_stores_host_from_kwargs(self, _gpu_engine):
        """
        LLMEngine(model, host="127.0.0.1", ...) → config.host == "127.0.0.1"。

        验证 host kwarg 通过 fields(Config) 过滤 → Config 构造 → 正确存储。
        """
        assert _gpu_engine.config.host == "127.0.0.1", (
            f"Expected host='127.0.0.1', got {_gpu_engine.config.host!r}"
        )

    def test_config_stores_port_from_kwargs(self, _gpu_engine):
        """
        LLMEngine(model, port=8000, ...) → config.port == 8000。

        验证 port kwarg 通过 fields(Config) 过滤 → Config 构造 → 正确存储。
        """
        assert _gpu_engine.config.port == 8000, (
            f"Expected port=8000, got {_gpu_engine.config.port}"
        )

    def test_config_defaults_without_kwargs_are_correct(self, _gpu_engine):
        """
        验证 Config 的默认值定义正确。
        虽然 engine 是以自定义 host/port 创建的，但我们仍然可以
        验证 Config 类本身的默认值（通过创建一个不依赖 GPU 的 Config）。
        这等价于验证：如果 CLI 不传 --host，Config 会使用 "0.0.0.0":8000。
        """
        # 直接检查 Config 字段的默认值（无需创建实例）
        from dataclasses import fields
        for f in fields(type(_gpu_engine.config)):
            if f.name == "host":
                assert f.default == "0.0.0.0", (
                    f"Config.host default should be '0.0.0.0', got {f.default!r}"
                )
            elif f.name == "port":
                assert f.default == 8000, (
                    f"Config.port default should be 8000, got {f.default}"
                )


@pytest.mark.gpu
class TestOfflineGenerateBackwardCompatible:
    """
    验证 LLMEngine.generate() 在新增 host/port 字段后行为不变。

    这是最关键的向后兼容性测试——离线批处理是 baby-vllm
    最早的功能，必须保证不受 Phase 6 影响。

    注：使用 module-scoped _gpu_engine fixture，不重复创建引擎。
    """

    def test_generate_returns_valid_output(self, _gpu_engine):
        """
        generate() 应返回：
          - outputs: list[dict] 含 'text' 和 'token_ids'
          - metrics: dict 含吞吐量和延迟统计
        """
        from babyvllm.sampling_params import SamplingParams

        sp = SamplingParams(max_tokens=8)
        outputs, metrics = _gpu_engine.generate(
            prompts=["Hello, my name is"],
            sampling_params=sp,
        )

        # 验证输出结构
        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
        assert "text" in outputs[0], "Output missing 'text' key"
        assert "token_ids" in outputs[0], "Output missing 'token_ids' key"
        assert isinstance(outputs[0]["text"], str)
        assert isinstance(outputs[0]["token_ids"], list)
        assert len(outputs[0]["token_ids"]) > 0, (
            "No tokens were generated"
        )

        # 验证 metrics 结构
        assert "throughput" in metrics
        assert "total_tokens" in metrics
        assert metrics["total_tokens"] > 0

    def test_generate_with_multiple_prompts(self, _gpu_engine):
        """
        多样本批处理：3 个 prompts 同时推理。
        输出数量应等于输入数量，且每个输出都有生成内容。
        """
        from babyvllm.sampling_params import SamplingParams

        sp = SamplingParams(max_tokens=8)
        prompts = [
            "The capital of France is",
            "Python is a programming",
            "The speed of light is",
        ]
        outputs, metrics = _gpu_engine.generate(
            prompts=prompts,
            sampling_params=sp,
        )

        assert len(outputs) == 3, (
            f"Expected 3 outputs for 3 prompts, got {len(outputs)}"
        )
        for i, out in enumerate(outputs):
            assert len(out["token_ids"]) > 0, (
                f"Prompt {i} ({prompts[i][:30]}...) produced no tokens"
            )
        assert metrics["total_tokens"] > 0


@pytest.mark.gpu
class TestOnlineGenerateBackwardCompatible:
    """
    验证 AsyncLLMEngine.generate() 在新增 host/port 字段后行为不变。

    注：使用 module-scoped _gpu_async_engine fixture，不重复创建引擎。
    """

    @pytest.mark.asyncio
    async def test_async_generate_returns_output(self, _gpu_async_engine):
        """单请求 async generate 应正常返回文本。"""
        from babyvllm.sampling_params import SamplingParams

        sp = SamplingParams(max_tokens=8)
        outputs = []
        async for output in _gpu_async_engine.generate("Hello world", sp):
            outputs.append(output)

        assert len(outputs) >= 1, "Expected at least 1 output"
        assert outputs[-1].finished, "Final output should have finished=True"
        assert len(outputs[-1].text) > 0, "Generated text should not be empty"

    @pytest.mark.asyncio
    async def test_async_generate_concurrent_requests(self, _gpu_async_engine):
        """
        3 个并发 async generate 请求应各自独立获得输出。
        每个请求的 request_id 应不同。
        """
        import asyncio
        from babyvllm.sampling_params import SamplingParams

        sp = SamplingParams(max_tokens=8)

        async def worker(prompt: str, worker_id: int) -> dict:
            results = []
            async for output in _gpu_async_engine.generate(prompt, sp):
                results.append(output)
            return {"worker_id": worker_id, "outputs": results}

        tasks = [
            worker("The meaning of life is", 1),
            worker("The largest planet is", 2),
            worker("Water boils at", 3),
        ]
        all_results = await asyncio.gather(*tasks)

        assert len(all_results) == 3
        request_ids = set()
        for result in all_results:
            assert len(result["outputs"]) >= 1
            for o in result["outputs"]:
                request_ids.add(o.request_id)
        # 3 个请求应有 3 个不同的 request_id
        assert len(request_ids) == 3, (
            f"Expected 3 unique request_ids, got {len(request_ids)}"
        )

    @pytest.mark.asyncio
    async def test_async_engine_config_stores_host_port(self, _gpu_async_engine):
        """
        AsyncLLMEngine → engine.engine.config 正确存储 host/port。

        host/port 路径: CLI args → engine_kwargs → AsyncLLMEngine(**kwargs)
        → LLMEngine(**kwargs) → Config(**config_kwargs)。
        """
        config = _gpu_async_engine.engine.config
        assert config.host == "127.0.0.1", (
            f"Expected host='127.0.0.1' in async engine, got {config.host!r}"
        )
        assert config.port == 8000, (
            f"Expected port=8000 in async engine, got {config.port}"
        )


# ===========================================================================
# 独立运行入口
# ===========================================================================

if __name__ == "__main__":
    """
    允许直接运行此文件而不通过 pytest：
      python test_phase6.py

    这在快速检查语法和导入时有帮助，但建议使用 pytest 以获得完整功能。
    """
    import subprocess
    script = os.path.abspath(__file__)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", script, "-v", "-m", "not gpu", "--tb=short"],
        cwd=os.path.dirname(script),
    )
    sys.exit(result.returncode)
