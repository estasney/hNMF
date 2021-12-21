from rich.live import Live
from rich.tree import Tree
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union, Optional


class ProgressTree:
    def __init__(self):
        self.live = None
        self.tree = None
        self.branches = {}

    def __enter__(self):
        self.tree = Tree("", guide_style="bold blue", hide_root=True)
        self._lookup_branch("Root", None, None)
        self.live = Live(self.tree)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.live.stop()

    def _lookup_branch(
        self, k: "Union[str, int]", source: "Optional[Tree]", desc: "Optional[int]"
    ) -> "Tree":
        branch = self.branches.get(k, None)
        if branch:
            return branch
        display_name = f"[green]{k}" if not desc else f"[green]{k}:({desc})"
        if source:
            branch = source.add(display_name)
        else:
            branch = self.tree.add(display_name)
        self.branches[k] = branch
        return branch

    def add_branch(
        self,
        source: "Union[str, int]",
        target: "Union[int, str]",
        desc: "Optional[int]",
    ):
        source_branch = self._lookup_branch(source, None, None)
        self._lookup_branch(target, source_branch, desc)
        self.live.update(self.tree)
