(define (problem tireworld-042-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l11) (road l0 l15) (road l1 l21) (road l1 l32) (road l10 l13) (road l10 l24) (road l11 l0) (road l11 l8) (road l12 l16) (road l12 l3) (road l13 l10) (road l13 l32) (road l14 l20) (road l14 l41) (road l15 l0) (road l15 l22) (road l16 l12) (road l16 l38) (road l17 l23) (road l17 l26) (road l18 l21) (road l18 l4) (road l19 l28) (road l19 l3) (road l2 l36) (road l2 l6) (road l20 l14) (road l20 l40) (road l21 l1) (road l21 l18) (road l22 l15) (road l22 l25) (road l23 l17) (road l23 l4) (road l24 l10) (road l24 l26) (road l25 l22) (road l25 l7) (road l26 l17) (road l26 l24) (road l26 l8) (road l27 l29) (road l27 l35) (road l28 l19) (road l28 l33) (road l28 l7) (road l29 l27) (road l29 l5) (road l3 l12) (road l3 l19) (road l30 l9) (road l31 l4) (road l31 l40) (road l32 l1) (road l32 l13) (road l33 l28) (road l33 l34) (road l34 l33) (road l34 l40) (road l35 l27) (road l35 l4) (road l36 l2) (road l36 l37) (road l37 l36) (road l37 l39) (road l38 l16) (road l38 l39) (road l39 l37) (road l39 l38) (road l4 l18) (road l4 l23) (road l4 l31) (road l4 l35) (road l40 l20) (road l40 l31) (road l40 l34) (road l41 l14) (road l41 l9) (road l5 l29) (road l5 l6) (road l6 l2) (road l6 l5) (road l7 l25) (road l7 l28) (road l8 l11) (road l8 l26) (road l9 l30) (road l9 l41) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l30))
    (:goal (vehicle-at l26))
)