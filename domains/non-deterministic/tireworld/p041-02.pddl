(define (problem tireworld-041-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l33) (road l0 l38) (road l1 l12) (road l1 l19) (road l10 l11) (road l10 l7) (road l11 l10) (road l11 l20) (road l11 l25) (road l12 l1) (road l12 l37) (road l13 l15) (road l13 l29) (road l14 l19) (road l14 l21) (road l15 l13) (road l15 l31) (road l16 l3) (road l16 l32) (road l17 l26) (road l17 l3) (road l18 l28) (road l18 l7) (road l19 l1) (road l19 l14) (road l19 l34) (road l2 l28) (road l2 l5) (road l20 l11) (road l20 l22) (road l21 l14) (road l21 l40) (road l22 l20) (road l22 l36) (road l22 l4) (road l23 l29) (road l23 l30) (road l23 l36) (road l24 l27) (road l24 l6) (road l25 l11) (road l25 l39) (road l26 l17) (road l26 l28) (road l27 l24) (road l27 l35) (road l27 l37) (road l28 l18) (road l28 l2) (road l28 l26) (road l29 l13) (road l29 l23) (road l3 l16) (road l3 l17) (road l3 l8) (road l30 l23) (road l30 l5) (road l31 l15) (road l31 l34) (road l32 l16) (road l32 l35) (road l33 l0) (road l33 l39) (road l34 l19) (road l34 l31) (road l35 l27) (road l35 l32) (road l36 l22) (road l36 l23) (road l37 l12) (road l37 l27) (road l37 l9) (road l38 l0) (road l38 l40) (road l39 l25) (road l39 l33) (road l4 l22) (road l4 l6) (road l40 l21) (road l40 l38) (road l5 l2) (road l5 l30) (road l6 l24) (road l6 l4) (road l7 l10) (road l7 l18) (road l8 l3) (road l8 l9) (road l9 l37) (road l9 l8) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l9) (vehicle-at l2))
    (:goal (vehicle-at l8))
)