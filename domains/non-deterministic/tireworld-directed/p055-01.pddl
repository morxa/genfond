(define (problem tireworld-055-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l6 l7 l8 l9 - location)
    (:init (road l0 l8) (road l1 l41) (road l1 l52) (road l10 l29) (road l11 l37) (road l12 l21) (road l13 l26) (road l13 l37) (road l14 l6) (road l15 l34) (road l16 l7) (road l17 l28) (road l17 l33) (road l18 l4) (road l18 l45) (road l19 l32) (road l2 l50) (road l21 l46) (road l22 l20) (road l22 l43) (road l23 l5) (road l24 l26) (road l24 l50) (road l25 l30) (road l26 l13) (road l26 l24) (road l26 l40) (road l26 l49) (road l27 l4) (road l27 l7) (road l28 l2) (road l29 l10) (road l29 l52) (road l3 l16) (road l3 l34) (road l30 l25) (road l30 l31) (road l31 l0) (road l31 l30) (road l32 l9) (road l33 l17) (road l34 l15) (road l34 l3) (road l35 l33) (road l36 l50) (road l37 l11) (road l37 l25) (road l38 l16) (road l39 l48) (road l4 l18) (road l4 l27) (road l40 l26) (road l41 l44) (road l42 l23) (road l43 l22) (road l43 l50) (road l44 l14) (road l44 l41) (road l45 l18) (road l45 l39) (road l46 l51) (road l47 l19) (road l48 l10) (road l49 l38) (road l49 l42) (road l5 l23) (road l5 l47) (road l50 l24) (road l50 l43) (road l51 l40) (road l51 l46) (road l52 l1) (road l52 l29) (road l53 l15) (road l53 l9) (road l54 l11) (road l54 l6) (road l6 l54) (road l7 l16) (road l7 l27) (road l8 l36) (road l9 l35) (road l9 l53) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l12))
    (:goal (vehicle-at l20))
)