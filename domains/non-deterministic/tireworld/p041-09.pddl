(define (problem tireworld-041-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l1) (road l1 l0) (road l1 l16) (road l10 l2) (road l10 l5) (road l11 l28) (road l11 l6) (road l12 l17) (road l12 l32) (road l13 l28) (road l13 l33) (road l13 l8) (road l14 l16) (road l14 l20) (road l14 l28) (road l15 l39) (road l15 l6) (road l16 l1) (road l16 l14) (road l16 l37) (road l17 l12) (road l17 l38) (road l18 l35) (road l18 l7) (road l19 l32) (road l19 l9) (road l2 l10) (road l2 l26) (road l2 l27) (road l20 l14) (road l20 l34) (road l21 l37) (road l21 l5) (road l22 l29) (road l22 l4) (road l22 l40) (road l23 l29) (road l23 l31) (road l24 l39) (road l24 l7) (road l25 l9) (road l26 l2) (road l26 l30) (road l27 l2) (road l27 l34) (road l28 l11) (road l28 l13) (road l28 l14) (road l28 l3) (road l29 l22) (road l29 l23) (road l3 l28) (road l3 l33) (road l3 l36) (road l30 l26) (road l30 l35) (road l31 l23) (road l31 l36) (road l32 l12) (road l32 l19) (road l33 l13) (road l33 l3) (road l34 l20) (road l34 l27) (road l35 l18) (road l35 l30) (road l36 l3) (road l36 l31) (road l37 l16) (road l37 l21) (road l38 l17) (road l38 l4) (road l39 l15) (road l39 l24) (road l4 l22) (road l4 l38) (road l40 l22) (road l40 l8) (road l5 l10) (road l5 l21) (road l6 l11) (road l6 l15) (road l7 l18) (road l7 l24) (road l8 l13) (road l8 l40) (road l9 l19) (road l9 l25) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l5) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l0))
    (:goal (vehicle-at l25))
)