
from rich.live import Live
from rich.tree import Tree


class ProgressTree:
    
    tree: Tree | None
    live : Live | None
    branches: dict[str | int, Tree]
    
    
    def __init__(self):
        self.live = None
        self.tree = None
        self.branches = {}

    def __enter__(self):
        self.tree = Tree("", guide_style="bold blue", hide_root=True)
        self._get_or_create_branch("Root", None, None)
        self.live = Live(self.tree)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.stop()

    def _get_or_create_branch(
        self, k: str | int, source: Tree | None, desc: int | None,
    ) -> "Tree":
        branch = self.branches.get(k, None)
        if branch:
            return branch
        display_name = f"[green]{k}" if not desc else f"[green]{k}:({desc})"
        if self.tree is None:
            raise RuntimeError("ProgressTree context not entered.")
        branch = source.add(display_name) if source else self.tree.add(display_name)
        self.branches[k] = branch
        return branch

    def add_branch(
        self,
        source: str | int,
        target: int | str,
        desc: int | None,
    ):
        
        if self.tree is None or self.live is None:
            raise RuntimeError("ProgressTree context not entered.")
        
        source_branch = self._get_or_create_branch(source, None, None)
        self._get_or_create_branch(target, source_branch, desc) 
        self.live.update(self.tree)
