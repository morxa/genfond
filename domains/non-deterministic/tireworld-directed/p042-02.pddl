(define (problem tireworld-042-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l17) (road l0 l23) (road l1 l18) (road l1 l32) (road l10 l22) (road l10 l8) (road l11 l13) (road l11 l20) (road l12 l15) (road l13 l11) (road l13 l37) (road l14 l4) (road l15 l39) (road l16 l32) (road l16 l38) (road l17 l0) (road l17 l5) (road l18 l1) (road l18 l26) (road l19 l12) (road l19 l41) (road l2 l33) (road l2 l37) (road l20 l11) (road l20 l35) (road l21 l24) (road l21 l38) (road l23 l0) (road l24 l21) (road l24 l27) (road l25 l36) (road l26 l18) (road l26 l40) (road l27 l24) (road l27 l6) (road l28 l6) (road l29 l31) (road l29 l7) (road l3 l36) (road l3 l40) (road l3 l5) (road l30 l34) (road l30 l36) (road l31 l29) (road l32 l1) (road l32 l16) (road l33 l2) (road l33 l28) (road l34 l14) (road l34 l30) (road l35 l20) (road l35 l39) (road l36 l25) (road l36 l3) (road l36 l30) (road l37 l13) (road l37 l2) (road l38 l16) (road l38 l21) (road l39 l15) (road l39 l35) (road l4 l10) (road l4 l14) (road l40 l26) (road l40 l3) (road l41 l19) (road l41 l7) (road l5 l17) (road l5 l3) (road l6 l27) (road l7 l29) (road l7 l41) (road l8 l10) (road l8 l25) (road l8 l9) (road l9 l31) (road l9 l8) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l20) (spare-in l21) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l41) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l23))
    (:goal (vehicle-at l22))
)