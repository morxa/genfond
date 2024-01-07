(define (problem tireworld-041-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l8) (road l1 l3) (road l1 l36) (road l10 l29) (road l10 l38) (road l11 l4) (road l12 l22) (road l13 l4) (road l13 l6) (road l14 l31) (road l14 l9) (road l15 l21) (road l15 l36) (road l16 l25) (road l16 l27) (road l17 l5) (road l18 l19) (road l18 l31) (road l19 l18) (road l2 l22) (road l2 l26) (road l20 l35) (road l21 l34) (road l22 l12) (road l22 l2) (road l23 l30) (road l24 l32) (road l24 l33) (road l25 l17) (road l26 l2) (road l26 l32) (road l27 l16) (road l27 l35) (road l28 l38) (road l28 l7) (road l29 l10) (road l29 l40) (road l3 l1) (road l30 l19) (road l30 l23) (road l30 l4) (road l31 l14) (road l31 l18) (road l32 l24) (road l32 l26) (road l33 l3) (road l34 l21) (road l34 l6) (road l35 l20) (road l35 l27) (road l36 l1) (road l36 l15) (road l37 l23) (road l38 l10) (road l38 l28) (road l39 l37) (road l4 l11) (road l40 l29) (road l40 l5) (road l5 l40) (road l6 l13) (road l7 l12) (road l7 l28) (road l8 l39) (road l9 l20) (spare-in l10) (spare-in l14) (spare-in l16) (spare-in l19) (spare-in l21) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l36) (spare-in l37) (spare-in l39) (spare-in l4) (spare-in l6) (spare-in l8) (vehicle-at l0))
    (:goal (vehicle-at l11))
)