-- Migration 026: «Тест-драйв» (pricing.TEST_PLAN, owner 2026-11-22: «в каждом режиме 10 разных обработок видео»).
-- One payment opens ONE user_subscriptions row PER video mode: plan_mode 'test:viral' | 'test:hooks' |
-- 'test:story' | 'test:edu' | 'test:streamer' | 'test:beep' (payment_store._activate_test_plan).
--   * plan_mode CHECK widened to those six values (cut / trailer / streamer unchanged);
--   * payment_id stays unique per (payment_id, plan_mode) instead of per payment (regular plans still open
--     exactly one row per payment, so nothing changes for them).
-- Debit / refund are the normal subscription path (charged_subscription_id, trigger 025). Idempotent.
-- Applying it does NOT put the plan on sale: checkout refuses test_all until SONYA_TEST_PLAN=on.
BEGIN;
ALTER TABLE user_subscriptions DROP CONSTRAINT IF EXISTS user_subscriptions_plan_mode_check;
ALTER TABLE user_subscriptions ADD CONSTRAINT user_subscriptions_plan_mode_check CHECK (plan_mode IN (
    'cut', 'trailer', 'streamer',
    'test:viral', 'test:hooks', 'test:story', 'test:edu', 'test:streamer', 'test:beep'));
ALTER TABLE user_subscriptions DROP CONSTRAINT IF EXISTS user_subscriptions_payment_id_key;
DROP INDEX IF EXISTS user_subscriptions_payment_mode_key;
CREATE UNIQUE INDEX user_subscriptions_payment_mode_key ON user_subscriptions (payment_id, plan_mode);
COMMIT;
