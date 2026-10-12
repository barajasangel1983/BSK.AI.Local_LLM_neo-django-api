class HistorianRouter:
    """Route the `historian` app to its databases.

    This keeps PLC historian data in Postgres while the rest of the
    application (chat, auth, etc.) continues to use SQLite. The models of the
    live historian (the PLC's rows, TimescaleDB) go to `historian_live`, which
    is read-only: no model writes there and nothing is migrated there.
    """

    app_label = "historian"
    live_db = "historian_live"

    def _db(self, model):
        if model._meta.app_label != self.app_label:
            return None
        return self.live_db if getattr(model, "live", False) else "historian"

    def db_for_read(self, model, **hints):
        return self._db(model)

    def db_for_write(self, model, **hints):
        return self._db(model)

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if db == self.live_db:
            return False
        if app_label == self.app_label:
            # Only migrate historian app on the historian DB
            return db == "historian"
        if db == "historian":
            # Block all other apps from migrating on the historian DB
            return False
        return None
