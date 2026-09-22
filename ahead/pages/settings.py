"""
ahead/pages/settings.py
=======================
Profile · Appearance · Preferences · Account — tabbed settings inspired by
a conventional account-settings layout.
"""

from html import escape

import streamlit as st

from ahead.components import current_user, logout, pref
from ahead.theme import APPEARANCE_OPTIONS, appearance_mode


def _row(icon: str, title: str, copy: str) -> str:
    return f"""
<div class="settings-row">
    <div class="settings-row-icon">{icon}</div>
    <div>
        <div class="settings-row-title">{escape(title)}</div>
        <div class="settings-row-copy">{escape(copy)}</div>
    </div>
</div>"""


def _panel(title: str, copy: str) -> None:
    st.html(f'<div class="settings-panel"><h2>{escape(title)}</h2><p>{escape(copy)}</p></div>')


def _sync_pref(name: str) -> None:
    st.session_state.prefs[name] = st.session_state[f"pref_widget_{name}"]


def _toggle_row(icon: str, title: str, copy: str, name: str) -> None:
    """A preference row. The toggle is fed from st.session_state.prefs and writes back on change."""
    left, right = st.columns([0.86, 0.14])
    with left:
        st.html(_row(icon, title, copy))
    with right:
        st.toggle(
            title, value=pref(name), key=f"pref_widget_{name}", label_visibility="collapsed",
            on_change=_sync_pref, args=(name,),
        )


# =============================================================================
# TABS
# =============================================================================

def _profile_tab(user: dict) -> None:
    _panel("Account information", "Review and update the information associated with your AHEAD profile.")
    st.html(
        f"""
<div class="settings-profile-banner">
    <div class="settings-avatar">{escape(user['initials'])}</div>
    <div>
        <div class="settings-person-name">{escape(user['name'])}</div>
        <div class="settings-person-role">{user['role_label']}</div>
        <div class="settings-person-email">{escape(user['email'])}</div>
    </div>
</div>
"""
    )
    with st.form("settings_profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("Full name", value=user["name"])
            st.text_input("Role", value=user["role_label"], disabled=True)
        with c2:
            email = st.text_input("Email", value=user["email"])
            institution = st.text_input(
                "Institution", value=user["institution"], disabled=not user["is_doctor"],
                placeholder="Not applicable for patient accounts" if not user["is_doctor"] else "",
            )
        if st.form_submit_button("Save changes", type="primary"):
            if not full_name.strip():
                st.error("Please enter your name.")
            elif "@" not in email or "." not in email.split("@")[-1]:
                st.error("Please enter a valid email address.")
            else:
                st.session_state.user.update(
                    {"name": full_name.strip(), "email": email.strip().lower(), "institution": institution.strip()}
                )
                st.toast("Profile updated.", icon="✅")
                st.rerun()


def _appearance_tab() -> None:
    _panel("Display mode", "Choose between a bright clinical interface and a darker low-light interface.")
    current = appearance_mode()
    columns = st.columns(3)
    for column, (mode, (css, copy)) in zip(columns, APPEARANCE_OPTIONS.items()):
        active = mode == current
        with column:
            st.html(
                f"""
<div class="appearance-choice {'active' if active else ''}">
    <div class="appearance-preview {css}"></div>
    <strong>{mode}{'<span class="check">✓</span>' if active else ''}</strong>
    <span>{copy}</span>
</div>
"""
            )
            if st.button(
                "Selected" if active else f"Use {mode}", key=f"appearance_{mode}", width="stretch",
                type="primary" if active else "secondary", disabled=active,
            ):
                st.session_state.appearance_mode = mode
                st.rerun()
    st.caption("The display mode applies to this browser session and switches charts and every AHEAD element immediately.")


def _preferences_tab() -> None:
    _panel("Notifications", "Choose which AHEAD updates you want to receive in this prototype.")
    with st.container(key="settings_rows"):
        _toggle_row("◉", "Screening results", "Notify when a new screening result is available.", "notify_screening")
        _toggle_row("▣", "System updates", "Important announcements about the platform.", "notify_system")
        _toggle_row("✎", "Educational content", "Tips and health-awareness information.", "notify_education")

    _panel("Data & privacy", "Control what AHEAD keeps during this browser session. Nothing is stored on a server.")
    with st.container(key="settings_rows_privacy"):
        _toggle_row("◈", "Keep screening history", "Show your session results on the Overview page.", "keep_history")
    history = st.session_state.get("screening_history", [])
    c1, c2 = st.columns([0.3, 0.7])
    with c1:
        if st.button(f"Clear session history ({len(history)})", disabled=not history, width="stretch"):
            st.session_state.screening_history = []
            st.session_state.last_results = {}
            st.toast("Screening history cleared.")
            st.rerun()

    _panel("Language", "English is currently available. Additional languages are planned.")
    st.selectbox("Application language", ["English"], disabled=True, key="pref_language")


def _account_tab(user: dict) -> None:
    _panel("Security & account", "Manage how you sign in to AHEAD and end your session.")

    with st.expander("🔒  Change password — update your password regularly"):
        with st.form("settings_password_form"):
            current_pw = st.text_input("Current password", type="password")
            new_pw = st.text_input("New password", type="password")
            confirm_pw = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Update password", type="primary"):
                if not current_pw or not new_pw:
                    st.error("Please fill in every field.")
                elif current_pw != (st.session_state.user or {}).get("password"):
                    st.error("The current password is incorrect.")
                elif len(new_pw) < 6:
                    st.error("Use at least 6 characters.")
                elif new_pw != confirm_pw:
                    st.error("The new passwords do not match.")
                else:
                    st.session_state.user["password"] = new_pw
                    st.success("Password updated for this session. (Demo accounts reset when the app restarts.)")

    with st.expander("🖥  Manage sessions — view and manage active sessions"):
        st.html(_row("●", "This browser", f"Signed in as {user['email']} · active now"))
        st.caption("AHEAD keeps no server-side sessions; signing out ends this one.")

    st.html(
        """
<div class="settings-danger">
    <strong>Log out</strong>
    <p>Sign out of your account. Your session screening history will be cleared.</p>
</div>
"""
    )
    if st.button("Log out", key="settings_logout", type="primary"):
        logout()

    st.html(
        """
<div class="settings-panel">
    <h2>About AHEAD</h2>
    <p>Advanced Health Early Awareness and Disease Detection — a Senior Design machine-learning screening
    prototype for diabetes, cardiovascular disease and chronic kidney disease. It provides educational
    screening estimates and does not diagnose disease.</p>
</div>
"""
    )


# =============================================================================
# PAGE
# =============================================================================

def render_settings() -> None:
    user = current_user()
    st.html(
        """
<div class="settings-shell-title">
    <h1>Settings</h1>
    <p>Manage your profile, display preferences, notifications and account.</p>
</div>
"""
    )
    with st.container(key="settings_tabs"):
        profile, appearance, preferences, account = st.tabs(["Profile", "Appearance", "Preferences", "Account"])
    with profile:
        _profile_tab(user)
    with appearance:
        _appearance_tab()
    with preferences:
        _preferences_tab()
    with account:
        _account_tab(user)
