"""File organization logic with support for Claude and Ollama backends."""

import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from .extractor import extract_preview
from .classifier import classify_bulk, validate_category as validate_claude_category, CATEGORIES as CLAUDE_CATEGORIES
from .ollama_classifier import classify_with_ollama, validate_category as validate_ollama_category, CATEGORIES as OLLAMA_CATEGORIES


class FileOrganizer:
    """Organizes files into categorized directories using AI classification."""

    def __init__(
        self,
        base_dir: Path,
        api_key: Optional[str] = None,
        dry_run: bool = False,
        skip_folders: bool = False,
        backend: str = "claude",
        ollama_model: str = "llama3.1:8b",
        ollama_url: str = "http://localhost:11434"
    ):
        """Initialize organizer.

        Args:
            base_dir: Base directory for categories (e.g., home directory)
            api_key: Anthropic API key (only for Claude backend)
            dry_run: If True, only print what would be done without moving files
            skip_folders: If True, ignore folders and only process individual files
            backend: "claude" or "ollama"
            ollama_model: Ollama model name to use (e.g., "llama3.1:8b")
            ollama_url: Ollama API URL (default: http://localhost:11434)
        """
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.api_key = api_key
        self.dry_run = dry_run
        self.skip_folders = skip_folders
        self.backend = backend
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.stats = {"files_processed": 0, "files_moved": 0, "folders_moved": 0, "errors": 0}

        # Select category system based on backend
        if backend == "claude":
            self.categories = CLAUDE_CATEGORIES
            self.validate_category = validate_claude_category
        else:
            self.categories = OLLAMA_CATEGORIES
            self.validate_category = validate_ollama_category

    def ensure_directories(self):
        """Create category directories if they don't exist."""
        for main, subs in self.categories.items():
            # Create main category
            (self.base_dir / main).mkdir(parents=True, exist_ok=True)
            # Create subcategories
            for sub in subs:
                (self.base_dir / main / sub).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory structure ready at {self.base_dir}")

    def scan_source(self, source_dir: Path) -> Tuple[List[Path], List[Path]]:
        """Scan source directory for files and folders to process.

        Returns:
            Tuple of (files, folders) - only top-level items
        """
        source = Path(source_dir).expanduser().resolve()

        # Get all items at top level only
        all_items = list(source.iterdir())

        # Separate files and directories
        files = [f for f in all_items if f.is_file()]
        folders = [f for f in all_items if f.is_dir()]

        # Filter out hidden files, images, and common non-document files
        skip_extensions = {'.tmp', '.cache', '.log', '.DS_Store',
                           '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg',
                           '.mp4', '.mov', '.avi', '.mkv', '.mp3', '.wav', '.aac',
                           '.zip', '.tar', '.gz', '.rar', '.7z',
                           '.exe', '.dmg', '.pkg', '.deb', '.rpm'}
        files = [f for f in files if f.suffix.lower() not in skip_extensions and not f.name.startswith('.')]

        # Filter out hidden folders
        folders = [f for f in folders if not f.name.startswith('.')]

        return files, folders

    def _classify_items(self, items_info: List[Dict], item_type: str = "files") -> List[Dict]:
        """Classify items using selected backend."""
        if self.backend == "claude":
            return classify_bulk(items_info, self.api_key)
        else:
            return classify_with_ollama(
                items_info,
                model=self.ollama_model,
                base_url=self.ollama_url,
                item_type=item_type
            )

    def process_batch(self, files: List[Path]) -> None:
        """Process a batch of files."""
        if not files:
            return

        print(f"\nProcessing batch of {len(files)} files...")

        # Extract previews
        files_info = []
        for i, filepath in enumerate(files):
            preview = extract_preview(filepath)
            files_info.append({
                "index": i,
                "filename": filepath.name,
                "preview": preview[:1000]  # Limit preview size
            })

        # Get classifications from AI
        try:
            classifications = self._classify_items(files_info, item_type="files")
        except Exception as e:
            print(f"Error calling API: {e}")
            classifications = []

        # Move files based on classifications
        for classification in classifications:
            self._apply_classification(files, classification)

    def _apply_classification(self, files: List[Path], classification: Dict) -> None:
        """Apply a single classification result."""
        try:
            idx = classification.get("index", 0)
            if idx >= len(files):
                return

            filepath = files[idx]
            category = classification.get("category", "Keep/Archives")
            reason = classification.get("reason", "No reason given")

            # Validate category
            main, sub = self.validate_category(category)

            # Determine destination
            if sub:
                dest_dir = self.base_dir / main / sub
                display_category = f"{main}/{sub}"
            else:
                dest_dir = self.base_dir / main
                display_category = main

            # Move file
            if self.dry_run:
                print(f"  [DRY RUN] Would move: {filepath.name} → {display_category}")
                print(f"             Reason: {reason}")
            else:
                dest_path = dest_dir / filepath.name

                # Handle duplicates
                counter = 1
                original_dest = dest_path
                while dest_path.exists():
                    stem = original_dest.stem
                    suffix = original_dest.suffix
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                shutil.move(str(filepath), str(dest_path))
                print(f"  ✓ {filepath.name} → {display_category}")
                print(f"    ({reason})")

            self.stats["files_moved"] += 1

        except Exception as e:
            print(f"  ✗ Error moving {filepath.name}: {e}")
            self.stats["errors"] += 1

    def process_folders(self, folders: List[Path]) -> None:
        """Process folders as units - classify and move entire folder."""
        if not folders:
            return

        print(f"\nFound {len(folders)} folder(s) to classify as units...")

        # Build folder info for classification
        folders_info = []
        for i, folder_path in enumerate(folders):
            # Get summary of folder contents
            contents = list(folder_path.iterdir())
            files_count = len([f for f in contents if f.is_file()])
            subdirs_count = len([f for f in contents if f.is_dir()])

            # Sample a few filenames for context
            sample_files = [f.name for f in contents[:5] if f.is_file()]
            sample_str = ", ".join(sample_files) if sample_files else "(no files)"

            folders_info.append({
                "index": i,
                "folder_name": folder_path.name,
                "files_count": files_count,
                "subdirs_count": subdirs_count,
                "sample_files": sample_str
            })

        # Classify folders
        try:
            classifications = self._classify_items(folders_info, item_type="folders")
        except Exception as e:
            print(f"Error classifying folders: {e}")
            classifications = []

        # Move folders based on classifications
        for classification in classifications:
            self._apply_folder_classification(folders, classification)

    def _apply_folder_classification(self, folders: List[Path], classification: Dict) -> None:
        """Apply classification to move an entire folder."""
        try:
            idx = classification.get("index", 0)
            if idx >= len(folders):
                return

            folder_path = folders[idx]
            category = classification.get("category", "Keep/Archives")
            reason = classification.get("reason", "No reason given")

            # Validate category
            main, sub = self.validate_category(category)

            # Determine destination
            if sub:
                dest_dir = self.base_dir / main / sub
                display_category = f"{main}/{sub}"
            else:
                dest_dir = self.base_dir / main
                display_category = main

            # Move folder
            if self.dry_run:
                print(f"  [DRY RUN] Would move folder: {folder_path.name}/ → {display_category}/")
                print(f"             Reason: {reason}")
            else:
                dest_path = dest_dir / folder_path.name

                # Handle duplicates
                counter = 1
                original_dest = dest_path
                while dest_path.exists():
                    dest_path = dest_dir / f"{folder_path.name}_{counter}"
                    counter += 1

                shutil.move(str(folder_path), str(dest_path))
                print(f"  ✓ {folder_path.name}/ → {display_category}/")
                print(f"    ({reason})")

            self.stats["folders_moved"] += 1

        except Exception as e:
            print(f"  ✗ Error moving folder {folder_path.name}: {e}")
            self.stats["errors"] += 1

    def process_directory(self, source_dir: Path, batch_size: int = 10) -> None:
        """Process all files and folders in a directory.

        Args:
            source_dir: Directory containing files/folders to organize
            batch_size: Number of files per API call
        """
        files, folders = self.scan_source(source_dir)

        if not files and not folders:
            print(f"No files or folders found in {source_dir}")
            return

        print(f"Found {len(files)} file(s) and {len(folders)} folder(s) to classify")
        print(f"Using backend: {self.backend}" + (f" ({self.ollama_model})" if self.backend == "ollama" else ""))

        if self.dry_run:
            print("[DRY RUN MODE - No files/folders will be moved]")

        # Process folders first (as units) - unless skipped
        if folders and not self.skip_folders:
            self.process_folders(folders)

        # Process individual files in batches
        if files:
            print(f"\n--- Processing {len(files)} individual files ---")
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(files) - 1) // batch_size + 1
                print(f"\n--- File Batch {batch_num}/{total_batches} ---")

                self.process_batch(batch)
                self.stats["files_processed"] += len(batch)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files moved:     {self.stats['files_moved']}")
        print(f"Folders moved:   {self.stats['folders_moved']}")
        print(f"Errors:          {self.stats['errors']}")
        print("=" * 40)
