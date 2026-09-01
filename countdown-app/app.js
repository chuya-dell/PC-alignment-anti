// Target Date: 2026-08-19 09:00:00 JST (UTC+9)
// Set using Date.UTC to guarantee 100% cross-platform compatibility (iOS/Safari)
const targetDate = new Date(Date.UTC(2026, 7, 19, 0, 0, 0));

// Threat messages array
const threatMessages = [
    "「まだ時間がある」と思ったあなた、過去問は何年分解きました？",
    "サイトを更新しても残り時間は増えませんよ。",
    "全落ちした後の言い訳、もう考え始めましたか？",
    "ライバルは今、あなたがこのメッセージを読んでいる間にも新しい知識を蓄えています。",
    "諦めて就活しますか？それともサボり続けて浪費しますか？",
    "あなたの「明日から本気出す」は、何回裏切られましたか？",
    "睡眠、スマホ、休憩。それらをすべて引いて、本当に合格できると信じていますか？",
    "合格発表の日、自分の番号がない画面を見つめる心の準備はできていますか？",
    "このタイマーが0になった瞬間、あなたの運命は決定されます。",
    "SNSを見る時間があるのに、公式を覚える時間はないのですか？",
    "「今日くらい休んでも大丈夫」——その妥協が不合格への第一歩です。",
    "ライバルはすでにあなたの遥か先を走っています。"
];

let currentMessageIndex = 0;

// DOM Elements
const dDays = document.getElementById('timer-days');
const dHours = document.getElementById('timer-hours');
const dMinutes = document.getElementById('timer-minutes');
const dSeconds = document.getElementById('timer-seconds');
const threatEl = document.getElementById('threat-message');

// Format numbers with padding
function pad(num) {
    return num.toString().padStart(2, '0');
}

// Main Countdown Update Logic
function updateCountdown() {
    const now = new Date();
    const diffMs = targetDate.getTime() - now.getTime();

    // 1. Exam period / post-exam check
    if (diffMs <= 0) {
        dDays.textContent = "00";
        dHours.textContent = "00";
        dMinutes.textContent = "00";
        dSeconds.textContent = "00";
        threatEl.textContent = "09:00 - 試験開始。積み重ねてきたもの（またはサボってきた成果）をそのまま答案に書いてきなさい。";
        return;
    }

    // 2. Normal countdown math
    const diffSeconds = Math.floor(diffMs / 1000);
    const days = Math.floor(diffSeconds / (3600 * 24));
    const hours = Math.floor((diffSeconds % (3600 * 24)) / 3600);
    const minutes = Math.floor((diffSeconds % 3600) / 60);
    const seconds = diffSeconds % 60;

    dDays.textContent = pad(days);
    dHours.textContent = pad(hours);
    dMinutes.textContent = pad(minutes);
    dSeconds.textContent = pad(seconds);

    // 3. Special Warning Messages based on Time
    const morningWindowMs = 3 * 60 * 60 * 1000; // 3 hours before (06:00 to 08:59 JST)
    if (diffMs <= morningWindowMs) {
        threatEl.textContent = "今さら焦っても過去のあなたのサボりは消えません。腹を括りなさい。";
    }
}

// Rotate general threat messages every 7 seconds (if not in special morning window)
function rotateMessage() {
    const now = new Date();
    const diffMs = targetDate.getTime() - now.getTime();
    
    // Do not rotate messages if within 3 hours of the exam or post-exam
    const morningWindowMs = 3 * 60 * 60 * 1000;
    if (diffMs <= morningWindowMs) {
        return;
    }

    threatEl.style.opacity = '0';
    setTimeout(() => {
        currentMessageIndex = (currentMessageIndex + 1) % threatMessages.length;
        threatEl.textContent = threatMessages[currentMessageIndex];
        threatEl.style.opacity = '1';
    }, 300);
}

// Visibility API handlers for tab title tracking
let defaultTitle = "【警告】現実直視カウントダウンシステム";
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        document.title = "[!] 逃げたな？戻れ";
    } else {
        document.title = "[WARNING] 戻ったか。早く勉強しろ。";
        setTimeout(() => {
            if (!document.hidden) {
                document.title = defaultTitle;
            }
        }, 3000);
    }
});

// Initialize Page Content
updateCountdown();
// Initial threat message
threatEl.textContent = threatMessages[0];

// Intervals
setInterval(updateCountdown, 1000);
setInterval(rotateMessage, 7000);
