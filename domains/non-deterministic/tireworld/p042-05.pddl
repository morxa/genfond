(define (problem tireworld-042-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l30) (road l0 l41) (road l1 l36) (road l1 l7) (road l10 l12) (road l10 l35) (road l11 l31) (road l11 l5) (road l12 l10) (road l12 l32) (road l13 l3) (road l13 l6) (road l14 l40) (road l14 l6) (road l15 l17) (road l15 l36) (road l16 l29) (road l16 l39) (road l17 l15) (road l17 l41) (road l18 l24) (road l18 l9) (road l19 l27) (road l2 l21) (road l2 l24) (road l20 l32) (road l20 l37) (road l21 l2) (road l21 l8) (road l22 l38) (road l22 l5) (road l23 l33) (road l23 l34) (road l24 l18) (road l24 l2) (road l25 l3) (road l25 l34) (road l26 l33) (road l27 l19) (road l27 l28) (road l27 l29) (road l28 l27) (road l28 l35) (road l29 l16) (road l29 l27) (road l3 l13) (road l3 l25) (road l30 l0) (road l30 l40) (road l31 l11) (road l31 l37) (road l32 l12) (road l32 l20) (road l33 l23) (road l33 l26) (road l33 l4) (road l34 l23) (road l34 l25) (road l35 l10) (road l35 l28) (road l36 l1) (road l36 l15) (road l37 l20) (road l37 l31) (road l38 l22) (road l38 l7) (road l39 l16) (road l39 l9) (road l4 l33) (road l4 l8) (road l40 l14) (road l40 l30) (road l41 l0) (road l41 l17) (road l5 l11) (road l5 l22) (road l6 l13) (road l6 l14) (road l7 l1) (road l7 l38) (road l8 l21) (road l8 l4) (road l9 l18) (road l9 l39) (spare-in l16) (spare-in l18) (spare-in l2) (spare-in l21) (spare-in l24) (spare-in l27) (spare-in l29) (spare-in l33) (spare-in l39) (spare-in l4) (spare-in l8) (spare-in l9) (vehicle-at l19))
    (:goal (vehicle-at l26))
)