-- PostgreSQL 数据库初始化脚本 (V3)
-- 用于智能饮食 Agent 的关系数据库

-- 创建数据库（如果不存在）
-- 注意：此命令需要在 psql 中以超级用户身份执行
-- CREATE DATABASE diet_agent_v3;

-- 连接到数据库
-- \c diet_agent_v3;

-- 1. 用户表
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- 2. 用户偏好表
CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    preference_type VARCHAR(50) NOT NULL,  -- 'health_goal', 'dietary_restriction', 'cuisine_preference', 'taste_preference'
    preference_value TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,  -- 偏好权重
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_type ON user_preferences(preference_type);

-- 3. 交互历史表
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    response TEXT,
    recommended_recipes JSONB,  -- 存储推荐的食谱 ID 列表
    context JSONB,  -- 存储交互时的上下文信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at);

-- 4. 反馈表
CREATE TABLE IF NOT EXISTS feedbacks (
    feedback_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    interaction_id INTEGER REFERENCES interactions(interaction_id) ON DELETE CASCADE,
    recipe_id VARCHAR(100) NOT NULL,  -- 对应 ChromaDB 中的食谱 ID
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),  -- 1-5 星评分
    feedback_type VARCHAR(50),  -- 'like', 'dislike', 'favorite', 'tried'
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_feedbacks_user_id ON feedbacks(user_id);
CREATE INDEX IF NOT EXISTS idx_feedbacks_recipe_id ON feedbacks(recipe_id);
CREATE INDEX IF NOT EXISTS idx_feedbacks_rating ON feedbacks(rating);

-- 5. 食材库存表
CREATE TABLE IF NOT EXISTS ingredient_inventory (
    inventory_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    ingredient_name VARCHAR(100) NOT NULL,
    quantity FLOAT,
    unit VARCHAR(20),
    expiry_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_ingredient_inventory_user_id ON ingredient_inventory(user_id);
CREATE INDEX IF NOT EXISTS idx_ingredient_inventory_ingredient ON ingredient_inventory(ingredient_name);
CREATE INDEX IF NOT EXISTS idx_ingredient_inventory_expiry ON ingredient_inventory(expiry_date);

-- 6. 营养目标表
CREATE TABLE IF NOT EXISTS nutrition_goals (
    goal_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    goal_type VARCHAR(50) NOT NULL,  -- 'daily_calories', 'protein', 'carbs', 'fat', 'fiber'
    target_value FLOAT NOT NULL,
    unit VARCHAR(20) NOT NULL,
    start_date DATE DEFAULT CURRENT_DATE,
    end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_nutrition_goals_user_id ON nutrition_goals(user_id);
CREATE INDEX IF NOT EXISTS idx_nutrition_goals_type ON nutrition_goals(goal_type);

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为需要自动更新 updated_at 的表创建触发器
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ingredient_inventory_updated_at
    BEFORE UPDATE ON ingredient_inventory
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_nutrition_goals_updated_at
    BEFORE UPDATE ON nutrition_goals
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 插入测试数据（可选）
-- 创建测试用户
INSERT INTO users (username) VALUES ('test_user') ON CONFLICT (username) DO NOTHING;

-- 添加测试用户偏好
INSERT INTO user_preferences (user_id, preference_type, preference_value, weight)
SELECT user_id, 'health_goal', '减肥', 1.0
FROM users WHERE username = 'test_user'
ON CONFLICT DO NOTHING;

INSERT INTO user_preferences (user_id, preference_type, preference_value, weight)
SELECT user_id, 'dietary_restriction', '低糖', 0.8
FROM users WHERE username = 'test_user'
ON CONFLICT DO NOTHING;

-- 添加测试食材库存
INSERT INTO ingredient_inventory (user_id, ingredient_name, quantity, unit, expiry_date)
SELECT user_id, '鸡胸肉', 500, 'g', CURRENT_DATE + INTERVAL '3 days'
FROM users WHERE username = 'test_user'
ON CONFLICT DO NOTHING;

INSERT INTO ingredient_inventory (user_id, ingredient_name, quantity, unit, expiry_date)
SELECT user_id, '西兰花', 300, 'g', CURRENT_DATE + INTERVAL '2 days'
FROM users WHERE username = 'test_user'
ON CONFLICT DO NOTHING;

-- 添加测试营养目标
INSERT INTO nutrition_goals (user_id, goal_type, target_value, unit)
SELECT user_id, 'daily_calories', 1800, 'kcal'
FROM users WHERE username = 'test_user'
ON CONFLICT DO NOTHING;

INSERT INTO nutrition_goals (user_id, goal_type, target_value, unit)
SELECT user_id, 'protein', 100, 'g'
FROM users WHERE username = 'test_user'
ON CONFLICT DO NOTHING;

-- 完成
COMMENT ON DATABASE diet_agent_v3 IS '智能饮食 Agent V3 数据库';
