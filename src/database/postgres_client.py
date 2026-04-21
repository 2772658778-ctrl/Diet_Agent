"""
PostgreSQL 客户端
用于存储用户信息、历史记录、反馈数据等结构化数据
"""

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import time
import uuid
from contextlib import contextmanager

from ..config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)
settings = get_settings()
_POSTGRES_CONNECT_TIMEOUT_SECONDS = 3
_POSTGRES_RETRY_COOLDOWN_SECONDS = 30.0


class PostgreSQLClient:
    """PostgreSQL 数据库客户端"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        初始化 PostgreSQL 客户端
        
        Args:
            connection_string: 数据库连接字符串
        """
        self.connection_string = connection_string or settings.postgres_connection_string
        self._connection = None

    def is_available(self) -> bool:
        """检测 PostgreSQL 是否可连接。"""
        conn = None
        try:
            conn = psycopg2.connect(
                self.connection_string,
                connect_timeout=_POSTGRES_CONNECT_TIMEOUT_SECONDS,
            )
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL 不可达，降级为无连接模式: {e}")
            return False
        finally:
            if conn is not None:
                conn.close()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = None
        try:
            conn = psycopg2.connect(
                self.connection_string,
                connect_timeout=_POSTGRES_CONNECT_TIMEOUT_SECONDS,
            )
            yield conn
            conn.commit()
        except Exception as e:
            if conn is not None:
                conn.rollback()
            logger.error(f"数据库操作失败: {e}", exc_info=True)
            raise
        finally:
            if conn is not None:
                conn.close()
    
    # ==================== 用户管理 ====================
    
    def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        height: Optional[float] = None,
        weight: Optional[float] = None,
        health_conditions: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None
    ) -> str:
        """
        创建用户
        
        Returns:
            user_id: 用户 ID
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                user_id = str(uuid.uuid4())
                
                cur.execute("""
                    INSERT INTO users (
                        user_id, username, email, age, gender, height, weight,
                        health_conditions, allergies
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING user_id
                """, (
                    user_id, username, email, age, gender, height, weight,
                    health_conditions or [], allergies or []
                ))
                
                result = cur.fetchone()
                logger.info(f"创建用户成功: {username} ({user_id})")
                return result[0]
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM users WHERE user_id = %s
                """, (user_id,))
                
                result = cur.fetchone()
                return dict(result) if result else None
    
    def update_user(self, user_id: str, **kwargs):
        """更新用户信息"""
        if not kwargs:
            return
        
        # 构建 SET 子句
        set_clause = ", ".join([f"{k} = %s" for k in kwargs.keys()])
        values = list(kwargs.values()) + [user_id]
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE users
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                """, values)
                
                logger.info(f"更新用户信息: {user_id}")
    
    # ==================== 用户偏好管理 ====================
    
    def set_user_preferences(
        self,
        user_id: str,
        favorite_tastes: Optional[List[str]] = None,
        disliked_tastes: Optional[List[str]] = None,
        favorite_cuisines: Optional[List[str]] = None,
        disliked_cuisines: Optional[List[str]] = None,
        favorite_ingredients: Optional[List[str]] = None,
        disliked_ingredients: Optional[List[str]] = None,
        health_goal: Optional[str] = None,
        target_calories: Optional[int] = None,
        target_protein: Optional[float] = None,
        max_cooking_time: Optional[int] = None,
        preferred_difficulty: Optional[str] = None
    ):
        """设置用户偏好"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_preferences (
                        user_id, favorite_tastes, disliked_tastes,
                        favorite_cuisines, disliked_cuisines,
                        favorite_ingredients, disliked_ingredients,
                        health_goal, target_calories, target_protein,
                        max_cooking_time, preferred_difficulty
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        favorite_tastes = EXCLUDED.favorite_tastes,
                        disliked_tastes = EXCLUDED.disliked_tastes,
                        favorite_cuisines = EXCLUDED.favorite_cuisines,
                        disliked_cuisines = EXCLUDED.disliked_cuisines,
                        favorite_ingredients = EXCLUDED.favorite_ingredients,
                        disliked_ingredients = EXCLUDED.disliked_ingredients,
                        health_goal = EXCLUDED.health_goal,
                        target_calories = EXCLUDED.target_calories,
                        target_protein = EXCLUDED.target_protein,
                        max_cooking_time = EXCLUDED.max_cooking_time,
                        preferred_difficulty = EXCLUDED.preferred_difficulty,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    user_id,
                    favorite_tastes or [],
                    disliked_tastes or [],
                    favorite_cuisines or [],
                    disliked_cuisines or [],
                    favorite_ingredients or [],
                    disliked_ingredients or [],
                    health_goal,
                    target_calories,
                    target_protein,
                    max_cooking_time,
                    preferred_difficulty
                ))
                
                logger.info(f"设置用户偏好: {user_id}")
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户偏好"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM user_preferences WHERE user_id = %s
                """, (user_id,))
                
                result = cur.fetchone()
                return dict(result) if result else None
    
    # ==================== 交互历史管理 ====================
    
    def log_interaction(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        agent_response: str,
        recommended_recipes: Optional[List[Dict]] = None,
        selected_recipe_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> str:
        """记录交互历史"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                interaction_id = str(uuid.uuid4())
                
                cur.execute("""
                    INSERT INTO interactions (
                        interaction_id, user_id, session_id,
                        user_input, agent_response,
                        recommended_recipes, selected_recipe_id, context
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING interaction_id
                """, (
                    interaction_id, user_id, session_id,
                    user_input, agent_response,
                    Json(recommended_recipes or []),
                    selected_recipe_id,
                    Json(context or {})
                ))
                
                # 更新用户统计
                cur.execute("""
                    UPDATE users
                    SET total_interactions = total_interactions + 1
                    WHERE user_id = %s
                """, (user_id,))
                
                logger.info(f"记录交互: {interaction_id}")
                return interaction_id
    
    def get_user_interactions(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取用户交互历史"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM interactions
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (user_id, limit, offset))
                
                results = cur.fetchall()
                return [dict(r) for r in results]

    def get_session_interactions(
        self,
        user_id: str,
        session_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if not user_id or not session_id:
            return []
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM interactions
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (user_id, session_id, limit))

                results = cur.fetchall()
                return [dict(r) for r in results]

    def list_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if not user_id:
            return []
        interactions = self.get_user_interactions(user_id, limit=max(limit * 10, 50), offset=0)
        session_map: dict[str, Dict[str, Any]] = {}
        ordered_session_ids: list[str] = []
        for item in interactions:
            session_id = str(item.get("session_id") or "").strip()
            if not session_id:
                continue
            if session_id not in session_map:
                session_map[session_id] = {
                    "session_id": session_id,
                    "interaction_count": 0,
                    "last_interaction_at": item.get("created_at"),
                    "last_query": str(item.get("user_input") or ""),
                    "last_response_preview": str(item.get("agent_response") or "")[:80],
                    "source": "postgres",
                }
                ordered_session_ids.append(session_id)
            session_map[session_id]["interaction_count"] += 1
        return [session_map[session_id] for session_id in ordered_session_ids[:limit]]
    
    # ==================== 反馈管理 ====================
    
    def add_feedback(
        self,
        user_id: str,
        recipe_id: str,
        rating: int,
        liked: bool,
        interaction_id: Optional[str] = None,
        taste_rating: Optional[int] = None,
        difficulty_rating: Optional[int] = None,
        time_accurate: Optional[bool] = None,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """添加反馈"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                feedback_id = str(uuid.uuid4())
                
                cur.execute("""
                    INSERT INTO feedbacks (
                        feedback_id, user_id, interaction_id, recipe_id,
                        rating, liked, taste_rating, difficulty_rating,
                        time_accurate, comment, tags
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING feedback_id
                """, (
                    feedback_id, user_id, interaction_id, recipe_id,
                    rating, liked, taste_rating, difficulty_rating,
                    time_accurate, comment, tags or []
                ))
                
                # 更新用户统计
                cur.execute("""
                    UPDATE users
                    SET total_feedbacks = total_feedbacks + 1
                    WHERE user_id = %s
                """, (user_id,))
                
                logger.info(f"添加反馈: {feedback_id}")
                return feedback_id
    
    def get_recipe_feedbacks(
        self,
        recipe_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取食谱反馈"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM feedbacks
                    WHERE recipe_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (recipe_id, limit))
                
                results = cur.fetchall()
                return [dict(r) for r in results]
    
    def get_user_feedbacks(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取用户反馈历史"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM feedbacks
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (user_id, limit))
                
                results = cur.fetchall()
                return [dict(r) for r in results]
    
    # ==================== 食材库存管理 ====================
    
    def add_ingredient_to_inventory(
        self,
        user_id: str,
        ingredient_name: str,
        quantity: float,
        unit: str,
        purchase_date: Optional[date] = None,
        expiry_date: Optional[date] = None
    ) -> str:
        """添加食材到库存"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                inventory_id = str(uuid.uuid4())
                
                cur.execute("""
                    INSERT INTO ingredient_inventory (
                        inventory_id, user_id, ingredient_name,
                        quantity, unit, purchase_date, expiry_date
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING inventory_id
                """, (
                    inventory_id, user_id, ingredient_name,
                    quantity, unit, purchase_date, expiry_date
                ))
                
                logger.info(f"添加食材到库存: {ingredient_name}")
                return inventory_id
    
    def get_user_inventory(
        self,
        user_id: str,
        status: Optional[str] = 'available'
    ) -> List[Dict[str, Any]]:
        """获取用户食材库存"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if status:
                    cur.execute("""
                        SELECT * FROM ingredient_inventory
                        WHERE user_id = %s AND status = %s
                        ORDER BY expiry_date ASC
                    """, (user_id, status))
                else:
                    cur.execute("""
                        SELECT * FROM ingredient_inventory
                        WHERE user_id = %s
                        ORDER BY expiry_date ASC
                    """, (user_id,))
                
                results = cur.fetchall()
                return [dict(r) for r in results]
    
    def update_inventory_status(self, inventory_id: str, status: str):
        """更新库存状态"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE ingredient_inventory
                    SET status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE inventory_id = %s
                """, (status, inventory_id))
                
                logger.info(f"更新库存状态: {inventory_id} -> {status}")


# 全局客户端实例
_postgres_client: Optional[PostgreSQLClient] = None
_postgres_retry_after_monotonic: float = 0.0


def get_postgres_client() -> Optional[PostgreSQLClient]:
    """获取 PostgreSQL 客户端实例（单例）"""
    global _postgres_client, _postgres_retry_after_monotonic
    now = time.monotonic()
    if _postgres_client is None and now < _postgres_retry_after_monotonic:
        return None
    
    if _postgres_client is None:
        candidate = PostgreSQLClient()
        if not candidate.is_available():
            _postgres_retry_after_monotonic = now + _POSTGRES_RETRY_COOLDOWN_SECONDS
            return None
        _postgres_client = candidate
        _postgres_retry_after_monotonic = 0.0
    
    return _postgres_client


# 兼容别名（enhanced_retriever_v3 使用此名称导入）
PostgresClient = PostgreSQLClient
