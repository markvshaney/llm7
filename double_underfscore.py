import os

def rename_python_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        # Rename _pycache_ directories
        if '_pycache_' in dirs:
            old_path = os.path.join(root, '_pycache_')
            new_path = os.path.join(root, '__pycache__')
            os.rename(old_path, new_path)
            
        # Rename files
        for file in files:
            if '_init_.py' in file:
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, file.replace('_init_', '__init__'))
                os.rename(old_path, new_path)
            elif '_init_.cpython' in file:
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, file.replace('_init_', '__init__'))
                os.rename(old_path, new_path)

# Run from your project root
rename_python_files('C:/Users/lbhei/source/repos/llm7')