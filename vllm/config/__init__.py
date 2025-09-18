# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.cache import CacheConfig
from vllm.config.compilation import (
    CompilationConfig,
    CompilationLevel,
    CUDAGraphMode,
    PassConfig,
)
from vllm.config.device import DeviceConfig
from vllm.config.kv_events import KVEventsConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.load import LoadConfig
from vllm.config.lora import LoRAConfig
from vllm.config.model import (
    ModelConfig,
    iter_architecture_defaults,
    try_match_architecture_defaults,
)
from vllm.config.multimodal import MultiModalConfig
from vllm.config.observability import ObservabilityConfig
from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.config.pooler import PoolerConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.config.structured_outputs import StructuredOutputsConfig
from vllm.config.utils import (
    ConfigType,
    SupportsMetricsInfo,
    config,
    get_attr_docs,
    is_init_field,
    update_config,
)
from vllm.config.vllm import (
    VllmConfig,
    get_cached_compilation_config,
    get_current_vllm_config,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)

# __all__ should only contain classes and functions.
# Types and globals should be imported from their respective modules.
__all__ = [
    # From vllm.config.cache
    "CacheConfig",
    # From vllm.config.compilation
    "CompilationConfig",
    "CompilationLevel",
    "CUDAGraphMode",
    "PassConfig",
    # From vllm.config.device
    "DeviceConfig",
    # From vllm.config.kv_events
    "KVEventsConfig",
    # From vllm.config.kv_transfer
    "KVTransferConfig",
    # From vllm.config.load
    "LoadConfig",
    # From vllm.config.lora
    "LoRAConfig",
    # From vllm.config.model
    "ModelConfig",
    "iter_architecture_defaults",
    "try_match_architecture_defaults",
    # From vllm.config.multimodal
    "MultiModalConfig",
    # From vllm.config.observability
    "ObservabilityConfig",
    # From vllm.config.parallel
    "EPLBConfig",
    "ParallelConfig",
    # From vllm.config.pooler
    "PoolerConfig",
    # From vllm.config.scheduler
    "SchedulerConfig",
    # From vllm.config.speculative
    "SpeculativeConfig",
    # From vllm.config.speech_to_text
    "SpeechToTextConfig",
    # From vllm.config.structured_outputs
    "StructuredOutputsConfig",
    # From vllm.config.utils
    "ConfigType",
    "SupportsMetricsInfo",
    "config",
    "get_attr_docs",
    "is_init_field",
    "update_config",
    # From vllm.config.vllm
    "VllmConfig",
    "get_cached_compilation_config",
    "get_current_vllm_config",
    "set_current_vllm_config",
    "get_layers_from_vllm_config",
]

@config
@dataclass
class AFDConfig:
    """Configuration for AFD (Attention FFN Disaggregation) distributed
    computation."""

    afd_connector: str = "dummy"
    """The AFD connector for vLLM to communicate between attention and FFN
    nodes. Available connectors: 'dummy', 'stepmesh'"""

    afd_role: Literal["attention", "ffn"] = "attention"
    """Role of this vLLM instance in AFD. 'attention' for attention workers,
    'ffn' for FFN servers."""

    afd_port: int = 1239
    """Port number for stepmesh parameter server communication."""

    afd_host: str = "127.0.0.1"
    """Host address for stepmesh parameter server communication."""

    num_afd_stages: int = 3
    """Number of pipeline stages for stage parallelism."""

    num_attention_servers: int = 1
    """Number of attention servers."""

    num_ffn_servers: int = 1
    """Number of FFN servers."""

    afd_server_rank: int = 0
    """Rank of this AFD server."""

    afd_extra_config: dict[str, Any] = field(default_factory=dict)
    """Extra configuration for specific AFD connectors."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # AFD configuration affects the computation graph structure
        # as it changes how FFN computation is performed
        factors: list[Any] = [
            self.afd_connector,
            self.afd_role,
            self.num_afd_stages,
            self.num_attention_servers,
            self.num_ffn_servers,
        ]
        
        return hashlib.sha256(str(factors).encode()).hexdigest()

    @property
    def is_attention_server(self) -> bool:
        """Check if this instance is configured as an attention server."""
        return self.afd_role == "attention"

    @property
    def is_ffn_server(self) -> bool:
        """Check if this instance is configured as an FFN server."""
        return self.afd_role == "ffn"


