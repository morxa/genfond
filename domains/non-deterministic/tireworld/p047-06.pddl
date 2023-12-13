(define (problem tireworld-047-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l5 l6 l7 l8 l9 - location)
    (:init (road l0 l46) (road l0 l9) (road l1 l3) (road l1 l4) (road l1 l6) (road l10 l18) (road l10 l3) (road l10 l37) (road l11 l15) (road l11 l22) (road l12 l24) (road l12 l35) (road l13 l18) (road l13 l2) (road l13 l21) (road l13 l28) (road l14 l27) (road l14 l38) (road l15 l11) (road l15 l27) (road l15 l39) (road l15 l9) (road l16 l38) (road l17 l30) (road l17 l8) (road l18 l10) (road l18 l13) (road l19 l34) (road l19 l35) (road l2 l13) (road l2 l40) (road l20 l26) (road l20 l8) (road l21 l13) (road l21 l4) (road l22 l11) (road l22 l31) (road l23 l26) (road l23 l43) (road l24 l12) (road l24 l41) (road l25 l42) (road l25 l43) (road l26 l20) (road l26 l23) (road l26 l41) (road l27 l14) (road l27 l15) (road l28 l13) (road l28 l29) (road l29 l28) (road l29 l37) (road l29 l44) (road l3 l1) (road l3 l10) (road l30 l17) (road l30 l32) (road l31 l22) (road l31 l33) (road l32 l30) (road l32 l5) (road l33 l31) (road l33 l9) (road l34 l19) (road l34 l7) (road l35 l12) (road l35 l19) (road l36 l45) (road l36 l5) (road l37 l10) (road l37 l29) (road l38 l14) (road l38 l16) (road l39 l15) (road l39 l7) (road l4 l1) (road l4 l21) (road l40 l2) (road l41 l24) (road l41 l26) (road l42 l25) (road l42 l6) (road l43 l23) (road l43 l25) (road l44 l29) (road l44 l46) (road l45 l36) (road l45 l9) (road l46 l0) (road l46 l44) (road l5 l32) (road l5 l36) (road l6 l1) (road l6 l42) (road l7 l34) (road l7 l39) (road l8 l17) (road l8 l20) (road l9 l0) (road l9 l15) (road l9 l33) (road l9 l45) (spare-in l0) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l44) (spare-in l46) (spare-in l7) (spare-in l9) (vehicle-at l16))
    (:goal (vehicle-at l40))
)