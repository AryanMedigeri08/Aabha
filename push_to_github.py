import subprocess
import sys

def run_cmd(args):
    print(f"\n> Running: {' '.join(args)}")
    try:
        res = subprocess.run(args, capture_output=True, text=True, shell=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        return res.returncode
    except Exception as e:
        print(f"Error: {e}")
        return -1

def main():
    print("=== Git Push Helper for Aabha ===")
    
    # 1. Run git status
    print("\nChecking git status...")
    run_cmd(["git", "status"])
    
    # 2. Ask for confirmation or proceed
    print("\nAdding files to git staging...")
    code = run_cmd(["git", "add", "ai-caption-generator/"])
    if code != 0:
        print("Failed to add files.")
        return
        
    print("\nCommitting changes...")
    code = run_cmd(["git", "commit", "-m", '"Implement accessible frontend UI, FastAPI static files serving, and deployment Dockerfile"'])
    if code != 0:
        print("Commit failed (might be no changes to commit).")
    
    print("\nPushing to GitHub remote repository...")
    code = run_cmd(["git", "push"])
    if code == 0:
        print("\nSuccessfully pushed to GitHub!")
    else:
        print("\nPush failed. Please ensure you have write permissions to the remote repository and are authenticated.")

if __name__ == "__main__":
    main()
