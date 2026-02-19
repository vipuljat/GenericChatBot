"""
Script to add default labels to existing plusminus/discrete questions
"""
from sqlalchemy.orm import Session, attributes
from stateful_services.database import get_db, SessionLocal
from stateful_services.db_schema import Question, Chatbot
from utils.logging import log

def fix_question_labels():
    """Add default labels to plusminus/discrete questions that don't have them"""
    db = SessionLocal()
    
    try:
        # Get all chatbots with quiz or people_analyzer mode
        chatbots = db.query(Chatbot).filter(
            Chatbot.mode.in_(["quiz", "people_analyzer"])
        ).all()
        
        updated_count = 0
        
        for chatbot in chatbots:
            log.info(f"Processing chatbot: {chatbot.chatbot_name}")
            
            # Get questions for this chatbot
            question_record = db.query(Question).filter(
                Question.chatbot_id == chatbot.chatbot_id
            ).first()
            
            if not question_record or not question_record.question_data:
                continue
            
            questions_data = question_record.question_data
            modified = False
            
            if isinstance(questions_data, list):
                for question in questions_data:
                    if isinstance(question, dict):
                        q_type = question.get("type", "")
                        
                        # Check if it's plusminus/discrete
                        if q_type in ["plusminus", "discrete"]:
                            labels = question.get("labels")
                            
                            # Add labels if missing or empty
                            if not labels or (isinstance(labels, list) and len(labels) == 0):
                                question["labels"] = ["Often", "Not Often", "Sometimes"]
                                modified = True
                                updated_count += 1
                                log.info(f"  ✓ Added labels to: {question.get('text', 'Unknown')[:60]}...")
            
            if modified:
                # Mark the field as modified to trigger SQLAlchemy update
                attributes.flag_modified(question_record, "question_data")
                db.commit()
                log.info(f"  ✓ Updated chatbot: {chatbot.chatbot_name}")
        
        log.info(f"\n✅ Complete! Updated {updated_count} questions.")
        
    except Exception as e:
        log.error(f"Error fixing labels: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🔧 Fixing question labels...")
    fix_question_labels()
    print("✅ Done!")
