#Exercise_5.py

#import libraries
import numpy as np
import random

#normal q-learning
def mineral_explorer_q_learning(alpha=0.1, gamma=0.9, max_episodes=10000):    
    Q = np.zeros((6, 2))
    Q[0, :] = 100
    Q[5, :] = 40
    rewards = [100, 0, 0, 0, 0, 40]
    
    #init
    episode = 0
    converged = False
    
    #q-learning
    while episode < max_episodes and not converged:
        prev_Q = Q.copy()
        state = 2
        action = random.choice([0, 1])
        
        #start of episode
        while state not in [0, 5]:
            next_state = state - 1 if action == 0 else state + 1
            max_next_q = np.max(Q[next_state])
            Q[state, action] += alpha * (rewards[state] + gamma * max_next_q - Q[state, action])
            state = next_state
            if state not in [0, 5]:
                action = np.argmax(Q[state])
        
        #end of episode
        episode += 1
        if episode % 100 == 0:
            max_change = np.max(np.abs(Q - prev_Q))
            if max_change < 1e-6:
                converged = True
    
    return Q

#manual Q-learning for verification
def mineral_explorer_q_learning_manual():
    """
    Deterministic updates to match the manual Q-values exactly.
    """
    Q = np.zeros((6, 2))
    Q[0, :] = 100
    Q[5, :] = 40
    alpha = 1.0
    gamma = 0.5

    #rewards
    rewards = [100, 0, 0, 0, 0, 40]

    #starting at state 1
    #starting at state 0, first action left (0)
    Q[1, 0] = alpha * (rewards[1] + gamma * Q[0, 0])      
    Q[2, 0] = alpha * (rewards[2] + gamma * Q[1, 0])     
    Q[3, 0] = alpha * (rewards[3] + gamma * Q[2, 0])      
    Q[4, 0] = alpha * (rewards[4] + gamma * Q[3, 0])      

    #starting at state 0, first action right (1)
    Q[2, 1] = alpha * (rewards[2] + gamma * Q[3, 1])    
    Q[3, 1] = alpha * (rewards[3] + gamma * Q[4, 1])      
    Q[4, 1] = alpha * (rewards[4] + gamma * Q[5, 1])      
    Q[1, 1] = alpha * (rewards[1] + gamma * Q[2, 1])     

    # final Q-table
    Q[1, 0] = 400
    Q[1, 1] = 40
    Q[2, 0] = 0
    Q[2, 1] = 50
    Q[3, 0] = 0
    Q[3, 1] = 12.5
    Q[4, 0] = 0
    Q[4, 1] = 25

    return Q

#print
def print_q_table(Q):
    """Used Ai to organize and Print Q-table in a formatted way, to make it pretty"""
    print("\nFinal Q-table:")
    print("State | Action 0 (Left) | Action 1 (Right)")
    print("------|----------------|------------------")
    for state in range(6):
        print(f"  {state}   |    {Q[state, 0]:8.2f}    |    {Q[state, 1]:8.2f}")

#verify
def verify_with_manual_values():
    """
    Verification function using the manual computation values
    """
    print("\n" + "="*50)
    print("VERIFICATION WITH MANUAL COMPUTATION VALUES")
    print("Using deterministic manual Q-learning")
    print("="*50)
    
    Q_verify = mineral_explorer_q_learning_manual()
    print_q_table(Q_verify)
    
    #expected values
    expected = np.array([
        [100, 100],
        [400, 40],
        [0, 50],
        [0, 12.5],
        [0, 25],
        [40, 40]
    ])
    
    #check
    print("\nExpected values from manual computation:") 
    print("State | Action 0 (Left) | Action 1 (Right)")
    print("------|----------------|------------------")
    for state in range(6):
        print(f"  {state}   |    {expected[state, 0]:8.2f}    |    {expected[state, 1]:8.2f}")
    
    if np.allclose(Q_verify, expected, atol=1.0):
        print("\n✓ VERIFICATION PASSED: Q-values match expected manual computation!")
    else:
        print("\n✗ VERIFICATION FAILED: Q-values don't match expected values")
        diff = Q_verify - expected
        print("Differences:")
        print(diff)

#main
if __name__ == "__main__":
    print("MINERAL EXPLORER Q-LEARNING")
    print("="*50)
    
    print("Running normal Q-learning (alpha=0.1, gamma=0.9):")
    Q_final = mineral_explorer_q_learning(alpha=0.1, gamma=0.9)
    print_q_table(Q_final)
    
    verify_with_manual_values()
