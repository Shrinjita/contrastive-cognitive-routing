# agent_test.py
from compare_runner import compare_all, print_results
import config

def main():
    query = "Should we approve the project funding request given current budget constraints?"
    
    results = compare_all(query)
    print_results(results)

if __name__ == "__main__":
    main()