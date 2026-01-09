"""
Script to remove duplicate responses from people_analyzer table.
Keeps only the most recent response for each reviewer-employee pair.
"""

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from stateful_services.db_schema import PeopleAnalyzer
from utils.logging import log
import config


def remove_duplicate_responses():
    """
    Find and remove duplicate responses in people_analyzer table.
    Keeps only the most recent response for each (chatbot_id, created_by, employee_id) combination.
    """
    
    # Create database connection
    engine = create_engine(config.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        log.info("🔍 Searching for duplicate responses...")
        
        # Find all duplicate combinations
        # Group by chatbot_id, created_by (reviewer), and employee_id (reviewee)
        duplicates = (
            db.query(
                PeopleAnalyzer.chatbot_id,
                PeopleAnalyzer.created_by,
                PeopleAnalyzer.employee_id,
                func.count(PeopleAnalyzer.id).label('count')
            )
            .group_by(
                PeopleAnalyzer.chatbot_id,
                PeopleAnalyzer.created_by,
                PeopleAnalyzer.employee_id
            )
            .having(func.count(PeopleAnalyzer.id) > 1)
            .all()
        )
        
        if not duplicates:
            log.info("✓ No duplicate responses found!")
            return
        
        log.info(f"Found {len(duplicates)} sets of duplicate responses")
        
        total_deleted = 0
        
        # For each duplicate set, keep only the most recent
        for chatbot_id, reviewer_id, employee_id, count in duplicates:
            log.info(f"Processing: Reviewer {reviewer_id} -> Employee {employee_id} ({count} entries)")
            
            # Get all reviews for this combination, ordered by created_at (newest first)
            reviews = (
                db.query(PeopleAnalyzer)
                .filter(
                    PeopleAnalyzer.chatbot_id == chatbot_id,
                    PeopleAnalyzer.created_by == reviewer_id,
                    PeopleAnalyzer.employee_id == employee_id
                )
                .order_by(PeopleAnalyzer.created_at.desc())
                .all()
            )
            
            # Keep the first (most recent), delete the rest
            if len(reviews) > 1:
                most_recent = reviews[0]
                duplicates_to_delete = reviews[1:]
                
                log.info(f"  Keeping: Review ID {most_recent.id} (created at {most_recent.created_at})")
                
                for duplicate_review in duplicates_to_delete:
                    log.info(f"  Deleting: Review ID {duplicate_review.id} (created at {duplicate_review.created_at})")
                    db.delete(duplicate_review)
                    total_deleted += 1
        
        # Commit the changes
        db.commit()
        
        log.info(f"✓ Successfully removed {total_deleted} duplicate responses")
        log.info(f"✓ Kept {len(duplicates)} most recent responses (one per reviewer-employee pair)")
        
    except Exception as e:
        log.error(f"❌ Error removing duplicates: {e}")
        db.rollback()
        raise
    
    finally:
        db.close()


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("DUPLICATE RESPONSE REMOVAL SCRIPT")
    log.info("=" * 60)
    
    try:
        remove_duplicate_responses()
        log.info("=" * 60)
        log.info("✓ Script completed successfully")
        log.info("=" * 60)
    except Exception as e:
        log.error(f"Script failed: {e}")
        log.info("=" * 60)
