import os
import sys
import importlib
import traceback

def print_debug_info():
    print("=" * 50)
    print("PYTHON ENVIRONMENT DEBUG INFORMATION")
    print("=" * 50)
    
    # Print current working directory
    print(f"Current Working Directory: {os.getcwd()}")
    
    # Print Python executable path
    print(f"Python Executable: {sys.executable}")
    
    # Print Python path
    print("\nPYTHON PATH:")
    for path in sys.path:
        print(path)
    
    # Print environment variables that might affect imports
    print("\nRELEVANT ENVIRONMENT VARIABLES:")
    env_vars = ['PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV']
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'NOT SET')}")
    
    # Attempt to list files in key directories
    print("\nDIRECTORY CONTENTS:")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"Project Root: {project_root}")
    
    try:
        print("\nAgents Directory Contents:")
        agents_dir = os.path.join(project_root, 'agents')
        print(os.listdir(agents_dir))
    except Exception as e:
        print(f"Error listing agents directory: {e}")
    
    # Attempt to import the problematic modules
    print("\nIMPORT DEBUGGING:")
    test_modules = [
        'agent_campaign', 
        'agents.agent_campaign', 
        'campaignai.agents.agent_campaign'
    ]
    
    for module_name in test_modules:
        print(f"\nTrying to import: {module_name}")
        try:
            imported_module = importlib.import_module(module_name)
            print(f"Successfully imported {module_name}")
            print(f"Module path: {imported_module.__file__}")
        except Exception as e:
            print(f"Import failed for {module_name}")
            print(traceback.format_exc())

if __name__ == "__main__":
    print_debug_info()