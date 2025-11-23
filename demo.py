#!/usr/bin/env python3
"""
Simple test and demo script for the Haiku Generator
"""

from haiku_generator import HaikuGenerator

def main():
    print("=" * 60)
    print("HAIKU GENERATOR - Quick Demo")
    print("=" * 60)
    print()
    
    # Initialize
    print("Initializing generator (this may take a few seconds)...")
    generator = HaikuGenerator('dataset.csv')
    
    print("\n" + "=" * 60)
    print("Generating 10 Sample Haikus")
    print("=" * 60)
    
    # Generate 10 haikus
    for i in range(10):
        print(f"\nHaiku #{i+1}:")
        print("-" * 40)
        haiku = generator.generate_haiku()
        print(haiku)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nTo run the interactive version, use:")
    print("  python haiku_generator.py")

if __name__ == "__main__":
    main()
