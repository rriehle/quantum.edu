# Create a structured curriculum outline for quantum computing education
import pandas as pd

curriculum_data = {
    'Module': [
        'Module 1: Mathematical Foundations',
        'Module 2: Quantum Mechanics Basics', 
        'Module 3: Quantum Computing Fundamentals',
        'Module 4: Quantum Programming Introduction',
        'Module 5: Core Quantum Algorithms',
        'Module 6: NISQ Programming and Applications',
        'Module 7: Quantum Machine Learning',
        'Module 8: Advanced Topics and Error Correction',
        'Module 9: Practical Projects and Implementation',
        'Module 10: Industry Applications and Future Directions'
    ],
    'Duration_Weeks': [2, 2, 3, 3, 4, 3, 4, 3, 4, 2],
    'Prerequisites': [
        'Basic programming knowledge',
        'Module 1 completed',
        'Modules 1-2 completed',
        'Modules 1-3 completed', 
        'Modules 1-4 completed',
        'Modules 1-5 completed',
        'Modules 1-6 completed',
        'Modules 1-7 completed',
        'Modules 1-8 completed',
        'All previous modules'
    ],
    'Difficulty_Level': [
        'Beginner',
        'Beginner-Intermediate',
        'Intermediate',
        'Intermediate',
        'Intermediate-Advanced',
        'Intermediate-Advanced', 
        'Advanced',
        'Advanced',
        'Advanced',
        'Expert'
    ]
}

curriculum_df = pd.DataFrame(curriculum_data)
print("Quantum Computing Curriculum Overview:")
print(curriculum_df.to_string(index=False))

# Calculate total duration
total_weeks = curriculum_df['Duration_Weeks'].sum()
print(f"\nTotal curriculum duration: {total_weeks} weeks (~{total_weeks//4} months)")

# Save to CSV for reference
curriculum_df.to_csv('quantum_curriculum_overview.csv', index=False)
print("\nCurriculum overview saved to quantum_curriculum_overview.csv")