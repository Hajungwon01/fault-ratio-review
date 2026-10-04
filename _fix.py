def sub(p, old, new):
    t=open(p,encoding='utf-8').read(); assert t.count(old)==1,(old[:60], t.count(old)); open(p,'w',encoding='utf-8').write(t.replace(old,new,1))
P='app/evaluation/scenarios.py'
sub(P,'''    reviewer_a: dict[str, int] = field(default_factory=dict)
    deviation: Optional[float] = None''','''    reviewer_a: dict[str, int] = field(default_factory=dict)
    reviewer_notes: dict[str, str] = field(default_factory=dict)  # 위원별 판단 근거 앞부분 (오답 원인 분석용, 재실행 없이 확인)
    deviation: Optional[float] = None
    spread: Optional[int] = None''')
sub(P,'''        reviewer_a={r.perspective: r.fault_ratio_a for r in report.reviews},
        deviation=report.deviation,''','''        reviewer_a={r.perspective: r.fault_ratio_a for r in report.reviews},
        reviewer_notes={r.perspective: r.rationale[:240] for r in report.reviews},
        deviation=report.deviation,
        spread=report.spread,''')
sub(P,'''            out.append(f"- {r.id}: {why}\n    위원 A과실 {r.reviewer_a} / 사실검증 지적 {r.fact_flagged[:2]} / 경고 {r.warnings}")''','''            notes = "".join(f"\n      · {p}({r.reviewer_a.get(p)}): {n}" for p, n in r.reviewer_notes.items())
            out.append(f"- {r.id}: {why}\n    위원 A과실 {r.reviewer_a} (편차 {r.deviation}, 최대-최소 {r.spread}) / 사실검증 지적 {r.fact_flagged[:2]} / 경고 {r.warnings}{notes}")''')
sub(P,'''        flag = "Y" if r.needs_precise_review else "n"
        lines.append(
            f"{r.id:<8}{r.difficulty:<5}{chart}{r.final_a:>6}{rng:>10}{inr}{(r.deviation if r.deviation is not None else 0):>6.1f}{flag:>6}"''','''        flag = "Y" if r.needs_precise_review else "n"
        lines.append(
            f"{r.id:<8}{r.difficulty:<5}{chart}{r.final_a:>6}{rng:>10}{inr}{(r.spread if r.spread is not None else 0):>6}{flag:>6}"''')
sub(P,'''    lines = [f"{'id':<8}{'난이도':<5}{'도표':<5}{'A과실':>6}{'정답범위':>10}{'범위':>5}{'편차':>6}{'플래그':>6}{'초':>6}{'호출':>5}  주 도표"]''','''    lines = [f"{'id':<8}{'난이도':<5}{'도표':<5}{'A과실':>6}{'정답범위':>10}{'범위':>5}{'위원차':>6}{'플래그':>6}{'초':>6}{'호출':>5}  주 도표"]''')
