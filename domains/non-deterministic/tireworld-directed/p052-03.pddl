(define (problem tireworld-052-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l6 l7 l8 l9 - location)
    (:init (road l0 l20) (road l1 l47) (road l10 l30) (road l10 l40) (road l11 l22) (road l11 l43) (road l12 l18) (road l12 l20) (road l12 l25) (road l13 l7) (road l14 l2) (road l14 l40) (road l15 l33) (road l16 l48) (road l17 l27) (road l17 l48) (road l18 l12) (road l18 l5) (road l19 l35) (road l2 l14) (road l2 l36) (road l20 l0) (road l20 l12) (road l21 l42) (road l22 l11) (road l23 l4) (road l24 l13) (road l24 l30) (road l25 l12) (road l26 l41) (road l26 l51) (road l27 l8) (road l28 l38) (road l28 l39) (road l28 l44) (road l29 l9) (road l3 l31) (road l30 l10) (road l30 l24) (road l30 l38) (road l31 l46) (road l32 l19) (road l32 l8) (road l33 l15) (road l33 l50) (road l34 l3) (road l34 l45) (road l35 l19) (road l35 l23) (road l36 l2) (road l36 l49) (road l37 l2) (road l38 l28) (road l39 l40) (road l4 l21) (road l40 l10) (road l40 l39) (road l41 l26) (road l42 l21) (road l42 l49) (road l43 l11) (road l43 l5) (road l44 l37) (road l45 l34) (road l45 l7) (road l46 l6) (road l47 l0) (road l47 l1) (road l48 l16) (road l48 l17) (road l49 l36) (road l49 l42) (road l5 l29) (road l5 l43) (road l50 l16) (road l50 l33) (road l51 l15) (road l6 l1) (road l6 l46) (road l7 l45) (road l8 l32) (road l9 l29) (road l9 l41) (spare-in l11) (spare-in l12) (spare-in l16) (spare-in l18) (spare-in l31) (spare-in l36) (spare-in l38) (spare-in l43) (spare-in l45) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l8) (vehicle-at l25))
    (:goal (vehicle-at l22))
)