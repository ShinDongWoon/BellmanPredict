def get_progress_bar(iterable=None, total=None, disable=False):
    """Return a tqdm progress bar or a no-op fallback.

    Parameters
    ----------
    iterable: optional
        Iterable to wrap with tqdm. If ``None``, a manual progress bar with the
        given ``total`` is returned.
    total: int, optional
        The total expected number of iterations. Only used when ``iterable`` is
        ``None`` or when passing it through to ``tqdm``.
    disable: bool, default False
        Whether to disable the progress bar.
    """
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        if iterable is not None:
            return iterable

        class _NoOpProgressBar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                pass

            def update(self, *args, **kwargs):
                pass

        return _NoOpProgressBar()

    if iterable is not None:
        return tqdm(iterable, total=total, disable=disable)
    return tqdm(total=total, disable=disable)
