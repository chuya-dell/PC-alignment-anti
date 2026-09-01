import os
import re
import string

def find_gdrive_root():
    """Find local Google Drive root directory on Windows."""
    for drive in string.ascii_uppercase:
        d_path = f"{drive}:/"
        if os.path.exists(d_path):
            for target in ["マイドライブ", "My Drive"]:
                full = os.path.join(d_path, target)
                if os.path.exists(full):
                    return full
    return None

def resolve_gdrive_path(path, create_if_missing=False):
    """
    Robustly resolves any drive letter or folder path to the local environment.
    Handles 'I:/マイドライブ/...' -> 'G:/マイドライブ/...' drive mapping,
    fuzzy folder matching, and output directory auto-creation.
    """
    if not path:
        return None

    # Normalize backslashes and quotes
    norm_path = path.replace('\\', '/').strip(' "\'')

    # 1. Direct match check
    if os.path.exists(norm_path):
        return os.path.abspath(norm_path)

    gdrive_root = find_gdrive_root()

    # 2. Match drive letter + 'マイドライブ' or 'My Drive'
    # e.g., "I:/マイドライブ/4.生データD_remo" -> subpath = "4.生データD_remo"
    match = re.match(r'^[A-Za-z]:/(?:マイドライブ|My Drive)/(.*)', norm_path, re.IGNORECASE)
    
    if gdrive_root and match:
        subpath = match.group(1)
        candidate = os.path.join(gdrive_root, subpath).replace('\\', '/')
        
        if os.path.exists(candidate):
            return candidate
            
        # Try fuzzy search in gdrive_root for matching folder name
        folder_name = os.path.basename(subpath.rstrip('/'))
        if folder_name:
            for root, dirs, _ in os.walk(gdrive_root):
                for d in dirs:
                    if d.lower() == folder_name.lower() or (len(folder_name) >= 3 and folder_name.lower() in d.lower()):
                        found = os.path.join(root, d).replace('\\', '/')
                        print(f"[PathResolver] Fuzzy matched '{path}' -> '{found}'")
                        return found
                        
        if create_if_missing:
            os.makedirs(candidate, exist_ok=True)
            print(f"[PathResolver] Created directory '{candidate}'")
            return candidate
            
    # 3. Create missing directory if requested
    if create_if_missing:
        if match and gdrive_root:
            subpath = match.group(1)
            candidate = os.path.join(gdrive_root, subpath).replace('\\', '/')
            os.makedirs(candidate, exist_ok=True)
            print(f"[PathResolver] Created directory '{candidate}'")
            return candidate
        else:
            os.makedirs(norm_path, exist_ok=True)
            print(f"[PathResolver] Created directory '{norm_path}'")
            return norm_path

    return norm_path

if __name__ == "__main__":
    # Test path resolution
    test_in = r"I:\マイドライブ\4.生データD_remo"
    test_out = r"I:\マイドライブ\5.解析結果_remo"
    print("Resolved In: ", resolve_gdrive_path(test_in))
    print("Resolved Out:", resolve_gdrive_path(test_out, create_if_missing=True))
