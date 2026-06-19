/**
 * Internal session label that mirrors the last-known Claude Code permission
 * mode for a claude-native session. The server stamps it from the
 * ``permission_mode`` field on every UserPromptSubmit / PreToolUse /
 * PostToolUse hook (see ``_observe_native_permission_mode``), so it survives
 * reconnect and seeds the badge before the first live ``session.mode`` event.
 */
const CLAUDE_NATIVE_PERMISSION_MODE_LABEL_KEY = "omnigent.claude_native.permission_mode";

/** Read the persisted permission mode off a session snapshot's labels. */
export function permissionModeFromSession(
  source: { labels?: Record<string, string | null> | null } | null | undefined,
): string | null {
  return source?.labels?.[CLAUDE_NATIVE_PERMISSION_MODE_LABEL_KEY] ?? null;
}

/** Label + Tailwind color classes for a permission-mode badge. Null means: don't render. */
export interface PermissionModeMeta {
  label: string;
  className: string;
}

export function permissionModeMeta(mode: string | null): PermissionModeMeta | null {
  switch (mode) {
    case "auto":
      return {
        label: "Auto mode",
        className:
          "border-green-300 bg-green-50 text-green-700 dark:border-green-500/30 dark:bg-green-500/10 dark:text-green-400",
      };
    case "plan":
      return {
        label: "Plan mode",
        className:
          "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-400",
      };
    case "acceptEdits":
      return {
        label: "Accept edits",
        className:
          "border-blue-300 bg-blue-50 text-blue-700 dark:border-blue-500/30 dark:bg-blue-500/10 dark:text-blue-400",
      };
    case "bypassPermissions":
      return {
        label: "Bypass permissions",
        className:
          "border-red-300 bg-red-50 text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400",
      };
    default:
      return null;
  }
}
