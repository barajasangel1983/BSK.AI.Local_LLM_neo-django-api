class HistorianRouter:
    """Route the `historian` app to the `historian` database.

    This keeps PLC historian data in Postgres while the rest of the
    application (chat, auth, etc.) continues to use SQLite.
    """

    app_label = "historian"

    def db_for_read(self, model, **hints):
        if model._meta.app_label == self.app_label:
            return "historian"
        return None

    def db_for_write(self, model, **hints):
        if model._meta.app_label == self.app_label:
            return "historian"
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label == self.app_label:
            # Only migrate historian app on the historian DB
            return db == "historian"
        if db == "historian":
            # Block all other apps from migrating on the historian DB
            return False
        return None
