# config_manager.py
import os
import yaml
from typing import Dict, Any, Optional

class ConfigManager:
    """Configuration manager for the chatbot application"""
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dictionary containing configuration values
        """
        # Create default config
        default_config = {
            'env_path': 'config/.env',
            'api_key_env_var': 'OPENAI_API_KEY_TEG',
            'app': {
                'title': 'AI Chatbot',
                'layout': 'wide'
            },
            'default_model': 'gpt-4o-mini',
            'available_models': ['gpt-4o-mini', 'gpt-3.5-turbo'],
            'default_system_message': 'Jesteś eksperckim asystentem. Odpowiadaj wyłącznie na podstawie dostarczonego KONTEKSTU. Jeśli kontekst nie zawiera odpowiedzi, odpowiedz: „Brak informacji w dostarczonym materiale.”',
            'app_title': 'My ChatBot',
            'chat_placeholder': 'Zadaj pytanie'
        }
        
        # Try to load config file
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    loaded_config = yaml.safe_load(file)
                
                # Merge loaded config with default config
                if loaded_config:
                    self._update_nested_dict(default_config, loaded_config)
            else:
                print(f"Config file not found at {self.config_path}, using default values")
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # Save default config
                with open(self.config_path, 'w') as file:
                    yaml.dump(default_config, file, default_flow_style=False)
                
        except Exception as e:
            print(f"Error loading config: {e}")
        
        return default_config
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary with values from another dictionary
        
        Args:
            d: Base dictionary to update
            u: Dictionary with values to update base dictionary
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration
        
        Returns:
            Dictionary containing all configuration values
        """
        return self.config
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value
        
        Args:
            key: The configuration key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The configuration value or default if not found
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save the configuration to the YAML file
        
        Args:
            config: Configuration dictionary to save (uses current config if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_to_save = config if config is not None else self.config
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as file:
                yaml.dump(config_to_save, file, default_flow_style=False)
            
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False