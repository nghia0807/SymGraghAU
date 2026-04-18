# ============================================================
# 1) AU–Expression combo rules (CNF)
#    (¬AU_i ∨ ¬AU_j ∨ ... ∨ Emotion_k)
#    (giữ nguyên như bạn đã chốt)
# ============================================================

CNF_AE_combo = [
    ["¬AU6", "¬AU12", "Happ"],
    ["¬AU1", "¬AU4", "Sad"],
    ["¬AU1", "¬AU2", "¬AU5", "¬AU25", "Surp"],
    ["¬AU4", "¬AU5", "¬AU25", "Fear"],
    ["¬AU4", "¬AU9", "Disg"],
]


# ============================================================
# 2) AU–AU Co-occurrence (logic gốc: AUm ∧ AUn)
#    → Lưu đúng semantics AND, KHÔNG chuyển sang OR.
#    Các cặp này thường xuất hiện cùng nhau trong FACS
#    (sad, fear/surprise, Duchenne smile, disgust...).
# ============================================================

AU_AA_cooccur = [
    ["AU1", "AU2"],    # nâng mày trong + ngoài (surprise/fear)
    ["AU1", "AU4"],    # buồn/sợ pha: inner raise + brow lowerer
    ["AU1", "AU5"],    # nâng mày + mở mắt (fear/surprise)
    ["AU1", "AU25"],   # biểu cảm căng thẳng / khóc: mày + môi mở

    ["AU2", "AU5"],    # outer raise + eye wide (surprise)

    ["AU4", "AU5"],    # brow lowerer + eye wide (anger/fear mix)
    ["AU4", "AU9"],    # cau mày + nhăn mũi (disgust-anger blend)

    ["AU5", "AU25"],   # mắt mở + môi mở (fear/surprise)
    ["AU9", "AU25"],   # disgust với miệng mở

    ["AU6", "AU12"],   # Duchenne smile (má + khóe môi)
    ["AU6", "AU25"],   # cười rộng, má nâng, môi mở
    ["AU12", "AU25"],  # cười hở răng (smile + lips part)
]


# ============================================================
# 3) AU–AU Mutual Exclusion (CNF: ¬AU_i ∨ ¬AU_j)
#    Các cặp cơ / biểu cảm thường đối nghịch nhau.
# ============================================================

CNF_AA_exclusion = [
    ["¬AU2", "¬AU4"],   # outer brow raise vs brow lowerer
    ["¬AU4", "¬AU12"],  # cau mày vs cười tươi
    ["¬AU9", "¬AU12"],  # disgust (nhăn mũi) vs enjoyment smile
    ["¬AU1", "¬AU9"],   # inner raise (sad/fear) vs nose wrinkle (disgust)
    ["¬AU2", "¬AU9"],   # outer raise (surprise) vs nose wrinkle (disgust)
    ["¬AU2", "¬AU6"],  # câu bạn trích trong paper
]
