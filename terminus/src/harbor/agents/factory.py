import importlib
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.models.agent.name import AgentName
from harbor.models.trial.config import AgentConfig


class AgentFactory:
    _AGENT_IMPORT_MAP: dict[AgentName, tuple[str, str]] = {
        AgentName.ORACLE: ("harbor.agents.oracle", "OracleAgent"),
        AgentName.NOP: ("harbor.agents.nop", "NopAgent"),
        AgentName.TERMINUS_2: ("harbor.agents.terminus_2", "Terminus2"),
        AgentName.TERMINUS_KIRA: ("harbor.agents.terminus_kira", "TerminusKira"),
    }

    @classmethod
    def create_agent_from_name(
        cls,
        name: AgentName,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs,
    ) -> BaseAgent:
        """
        Create an agent from an agent name.

        Args:
            name (AgentName): The name of the agent.

        Returns:
            BaseAgent: The created agent.

        Raises:
            ValueError: If the agent name is invalid.
        """
        if name not in cls._AGENT_IMPORT_MAP:
            raise ValueError(
                f"Unknown agent type: {name}. This could be because the agent is not "
                "registered in the AgentFactory or because the agent name is invalid."
            )

        module_path, class_name = cls._AGENT_IMPORT_MAP[name]
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)

        return agent_class(logs_dir=logs_dir, model_name=model_name, **kwargs)

    @classmethod
    def create_agent_from_import_path(
        cls,
        import_path: str,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs,
    ) -> BaseAgent:
        """
        Create an agent from an import path.

        Args:
            import_path (str): The import path of the agent. In the format
                'module.path:ClassName'.

        Returns:
            BaseAgent: The created agent.

        Raises:
            ValueError: If the import path is invalid.
        """
        if ":" not in import_path:
            raise ValueError("Import path must be in format 'module.path:ClassName'")

        module_path, class_name = import_path.split(":", 1)

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ValueError(f"Failed to import module '{module_path}': {e}") from e

        try:
            Agent = getattr(module, class_name)
        except AttributeError as e:
            raise ValueError(
                f"Module '{module_path}' has no class '{class_name}'"
            ) from e

        return Agent(logs_dir=logs_dir, model_name=model_name, **kwargs)

    @classmethod
    def create_agent_from_config(
        cls,
        config: AgentConfig,
        logs_dir: Path,
        **kwargs,
    ) -> BaseAgent:
        """
        Create an agent from an agent configuration.

        Args:
            config (AgentConfig): The configuration of the agent.

        Returns:
            BaseAgent: The created agent.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if config.name is not None and config.name in AgentName.values():
            return cls.create_agent_from_name(
                AgentName(config.name),
                logs_dir=logs_dir,
                model_name=config.model_name,
                extra_env=config.env,
                **config.kwargs,
                **kwargs,
            )
        elif config.import_path is not None:
            return cls.create_agent_from_import_path(
                config.import_path,
                logs_dir=logs_dir,
                model_name=config.model_name,
                extra_env=config.env,
                **config.kwargs,
                **kwargs,
            )
        elif config.name is not None:
            raise ValueError(
                f"Agent name {config.name} is not valid. Valid agent names: {
                    AgentName.values()
                }"
            )
        else:
            raise ValueError(
                "At least one of agent_name or agent_import_path must be set. "
                + f"Valid agent names: {AgentName.values()}"
            )
