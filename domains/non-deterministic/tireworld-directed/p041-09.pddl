(define (problem tireworld-041-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l9) (road l1 l13) (road l1 l35) (road l10 l3) (road l11 l26) (road l11 l32) (road l12 l19) (road l14 l34) (road l15 l2) (road l15 l29) (road l16 l20) (road l17 l8) (road l18 l10) (road l19 l21) (road l2 l16) (road l20 l37) (road l21 l16) (road l22 l4) (road l23 l12) (road l24 l40) (road l25 l23) (road l26 l27) (road l27 l13) (road l27 l31) (road l28 l11) (road l29 l6) (road l3 l15) (road l30 l33) (road l31 l27) (road l32 l22) (road l33 l14) (road l34 l25) (road l35 l27) (road l36 l30) (road l37 l39) (road l38 l5) (road l39 l24) (road l4 l38) (road l40 l17) (road l5 l18) (road l5 l7) (road l6 l36) (road l7 l31) (road l8 l0) (road l9 l1) (spare-in l10) (spare-in l11) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l30) (spare-in l32) (spare-in l36) (spare-in l38) (spare-in l4) (spare-in l5) (spare-in l6) (vehicle-at l28))
    (:goal (vehicle-at l13))
)