class EasyDict(dict):
    """Minimal drop-in replacement for easydict.EasyDict."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = self._convert(value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._convert(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setitem__(self, key, value):
        super().__setitem__(key, self._convert(value))

    @classmethod
    def _convert(cls, value):
        if isinstance(value, dict) and not isinstance(value, EasyDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(v) for v in value]
        return value

