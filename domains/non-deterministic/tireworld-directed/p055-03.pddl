(define (problem tireworld-055-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l6 l7 l8 l9 - location)
    (:init (road l0 l42) (road l1 l14) (road l1 l23) (road l10 l3) (road l10 l52) (road l10 l54) (road l11 l21) (road l12 l26) (road l12 l29) (road l12 l34) (road l13 l0) (road l13 l20) (road l14 l1) (road l15 l45) (road l16 l22) (road l16 l31) (road l17 l18) (road l18 l17) (road l18 l38) (road l19 l29) (road l19 l54) (road l19 l6) (road l2 l36) (road l20 l13) (road l20 l28) (road l21 l4) (road l22 l41) (road l23 l50) (road l24 l35) (road l24 l49) (road l25 l3) (road l25 l49) (road l25 l51) (road l26 l27) (road l26 l34) (road l26 l52) (road l27 l7) (road l28 l20) (road l28 l37) (road l28 l6) (road l29 l12) (road l3 l10) (road l3 l25) (road l30 l45) (road l30 l53) (road l31 l16) (road l31 l38) (road l32 l11) (road l32 l41) (road l33 l43) (road l34 l26) (road l35 l24) (road l35 l44) (road l35 l47) (road l36 l46) (road l37 l28) (road l38 l31) (road l39 l42) (road l39 l47) (road l4 l21) (road l4 l48) (road l40 l2) (road l41 l22) (road l41 l32) (road l42 l0) (road l42 l39) (road l43 l33) (road l43 l5) (road l44 l35) (road l45 l15) (road l45 l30) (road l46 l8) (road l47 l35) (road l47 l39) (road l48 l44) (road l49 l24) (road l49 l25) (road l5 l37) (road l50 l15) (road l51 l14) (road l52 l10) (road l52 l26) (road l53 l9) (road l54 l10) (road l54 l19) (road l6 l28) (road l7 l40) (road l8 l33) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l6) (spare-in l7) (spare-in l8) (vehicle-at l17))
    (:goal (vehicle-at l9))
)