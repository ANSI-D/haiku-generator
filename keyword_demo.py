#!/usr/bin/env python3
"""
Keyword-based Haiku Generation Demo
"""

from haiku_generator import HaikuGenerator

def main():
    print("=" * 60)
    print("KEYWORD-BASED HAIKU GENERATOR - Demo")
    print("=" * 60)
    print()
    
    # Initialize - models load from cache, should be fast
    print("Loading generator from cache...")
    generator = HaikuGenerator('dataset.csv')
    
    print("\n" + "=" * 60)
    print("Interactive Mode - Ready!")
    print("=" * 60)
    print("\nEnter a keyword to generate a haiku about that topic")
    print("Special commands:")
    print("  - 'stats' to see cache statistics")
    print("  - 'quit' or 'exit' to quit")
    
    while True:
        keyword = input("\nKeyword> ").strip()
        
        if keyword.lower() in ['quit', 'exit', 'q', 'stats']:
            if keyword.lower() == 'stats':
                # Show cache statistics
                stats = generator.get_cache_stats()
                print("\n" + "="*40)
                print("CACHE STATISTICS")
                print("="*40)
                print(f"Cached models: {stats['cached_models']}")
                print(f"Cache hits: {stats['cache_hits']}")
                print(f"Cache misses: {stats['cache_misses']}")
                print(f"Hit rate: {stats['hit_rate']:.1f}%")
                print(f"Total requests: {stats['total_requests']}")
                continue
            else:
                print("\nThanks for trying the keyword-based haiku generator!")
                print("\nFinal cache statistics:")
                stats = generator.get_cache_stats()
                print(f"  Cached models: {stats['cached_models']}")
                print(f"  Cache hit rate: {stats['hit_rate']:.1f}%")
                break
        
        if not keyword:
            print("Please enter a keyword (or 'stats' to see cache info).")
            continue
        
        haiku, score, breakdown = generator.generate_haiku_with_keyword(keyword, return_score=True)
        print(f"\n{haiku}")
        print(f"\nQuality Score: {score:.1f}/100")
        
        # Show score breakdown
        if breakdown:
            print(f"  Line Scores: {[f'{s:.1f}' for s in breakdown.get('line_scores', [])]}")
            print(f"  Coherence Bonus: {breakdown.get('coherence_bonus', 0):.1f}")
        
        is_valid, message = generator.verify_haiku_structure(haiku)
        print(f"\n{message}")

if __name__ == "__main__":
    main()
