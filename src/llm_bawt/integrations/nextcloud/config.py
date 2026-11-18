"""Nextcloud Talk bot configuration management.

Nextcloud bot configs are stored in the bot_profiles DB table under the
nextcloud_config JSONB column. This module uses the centralized bot config
loader from llm_bawt.bots.
"""

from dataclasses import dataclass
from typing import Optional

from llm_bawt.bots import (
    get_all_raw_bot_data,
    save_user_bot_config,
    remove_user_bot_section,
)


@dataclass
class NextcloudBot:
    """Configuration for a single Nextcloud Talk bot."""
    llm_bawt_bot: str  # The bot slug in bot_profiles
    nextcloud_bot_id: int
    secret: str
    conversation_token: str
    enabled: bool = True
    # Application user this room's mapping belongs to (enforced in a later slice).
    user_id: Optional[str] = None
    # Nextcloud user ID allowed in this room; required whenever user_id is set.
    actor_id: Optional[str] = None

    @property
    def name(self) -> str:
        """Bot name is the llm-bawt bot ID."""
        return self.llm_bawt_bot


class NextcloudBotConfig:
    """Manages Nextcloud bot configuration.

    Uses the centralized bot config loader which reads from the
    bot_profiles DB table. Config is persisted via BotProfileStore.
    """

    def __init__(self, bots_yaml_path=None):
        """Initialize config.
        
        Args:
            bots_yaml_path: Ignored - kept for backwards compatibility.
                           Config loading is now centralized in llm_bawt.bots.
        """
        # No separate loading needed - uses centralized loader
        pass

    def _get_nextcloud_bots(self) -> dict[str, NextcloudBot]:
        """Get all bots with nextcloud config from centralized loader."""
        bots = {}
        for slug, bot_data in get_all_raw_bot_data().items():
            nc_config = bot_data.get('nextcloud')
            if nc_config:
                bots[slug] = NextcloudBot(
                    llm_bawt_bot=slug,
                    nextcloud_bot_id=nc_config.get('bot_id'),
                    secret=nc_config.get('secret', ''),
                    conversation_token=nc_config.get('conversation_token'),
                    enabled=nc_config.get('enabled', True),
                    user_id=nc_config.get('user_id'),
                    actor_id=nc_config.get('actor_id'),
                )
        return bots

    @property
    def bots(self) -> dict[str, NextcloudBot]:
        """Get all configured Nextcloud bots."""
        return self._get_nextcloud_bots()

    def load(self):
        """Reload config - triggers centralized reload via property access."""
        # No-op: _get_nextcloud_bots() calls get_all_raw_bot_data() which 
        # checks for config file changes automatically
        pass

    def _check_reload(self):
        """Check for config changes - handled by centralized loader."""
        # No-op: handled automatically by get_all_raw_bot_data()
        pass

    def save(self):
        """Save is handled per-bot via add_bot() now."""
        # No-op: saves happen individually via save_user_bot_config()
        pass

    def add_bot(
        self,
        llm_bawt_bot: str,
        nextcloud_bot_id: int,
        secret: str,
        conversation_token: str,
    ) -> NextcloudBot:
        """Add Nextcloud config to a bot.
        
        Saves to bot_profiles DB table via BotProfileStore.
        """
        nc_data = {
            'bot_id': nextcloud_bot_id,
            'secret': secret,
            'conversation_token': conversation_token,
            'enabled': True,
        }
        save_user_bot_config(llm_bawt_bot, 'nextcloud', nc_data)
        
        return NextcloudBot(
            llm_bawt_bot=llm_bawt_bot,
            nextcloud_bot_id=nextcloud_bot_id,
            secret=secret,
            conversation_token=conversation_token,
            enabled=True,
        )

    def remove_bot(self, llm_bawt_bot: str) -> bool:
        """Remove Nextcloud config from a bot.
        
        Removes from user config only (doesn't touch repo).
        """
        return remove_user_bot_section(llm_bawt_bot, 'nextcloud')

    @staticmethod
    def _with_secret(bot: Optional[NextcloudBot]) -> Optional[NextcloudBot]:
        """Resolve the signing secret for one bot: encrypted record, else legacy plaintext."""
        if bot is not None:
            from .signing import resolve_secret
            bot.secret = resolve_secret(bot.llm_bawt_bot, bot.secret)
        return bot

    def get_bot_by_conversation(self, token: str) -> Optional[NextcloudBot]:
        """Get bot config by conversation token."""
        for bot in self.bots.values():
            if bot.enabled and bot.conversation_token == token:
                return self._with_secret(bot)
        return None

    def get_bot(self, llm_bawt_bot: str) -> Optional[NextcloudBot]:
        """Get bot config by llm-bawt bot ID."""
        return self._with_secret(self.bots.get(llm_bawt_bot))
