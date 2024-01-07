(define (problem tireworld-054-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l32) (road l1 l14) (road l1 l33) (road l10 l11) (road l11 l10) (road l11 l8) (road l12 l0) (road l12 l26) (road l13 l18) (road l13 l44) (road l14 l9) (road l15 l42) (road l16 l28) (road l16 l38) (road l17 l24) (road l17 l35) (road l18 l13) (road l18 l41) (road l19 l44) (road l2 l36) (road l2 l6) (road l2 l7) (road l20 l21) (road l20 l45) (road l21 l20) (road l21 l23) (road l22 l33) (road l23 l21) (road l24 l17) (road l24 l53) (road l25 l46) (road l26 l12) (road l26 l52) (road l27 l15) (road l27 l51) (road l28 l16) (road l28 l8) (road l29 l6) (road l3 l38) (road l3 l6) (road l30 l10) (road l30 l48) (road l31 l22) (road l31 l38) (road l31 l39) (road l32 l0) (road l32 l45) (road l33 l1) (road l34 l48) (road l34 l7) (road l35 l17) (road l35 l4) (road l36 l6) (road l37 l29) (road l37 l47) (road l38 l16) (road l38 l3) (road l38 l31) (road l39 l1) (road l39 l31) (road l4 l35) (road l4 l42) (road l40 l19) (road l40 l9) (road l41 l18) (road l41 l25) (road l42 l15) (road l42 l4) (road l43 l46) (road l43 l51) (road l44 l13) (road l44 l19) (road l45 l20) (road l45 l32) (road l46 l43) (road l47 l37) (road l47 l49) (road l48 l30) (road l48 l34) (road l49 l47) (road l49 l50) (road l5 l23) (road l5 l53) (road l50 l49) (road l50 l52) (road l51 l27) (road l51 l43) (road l52 l26) (road l52 l50) (road l53 l5) (road l6 l2) (road l6 l29) (road l6 l3) (road l7 l2) (road l7 l34) (road l8 l11) (road l8 l28) (road l9 l14) (road l9 l40) (spare-in l2) (spare-in l28) (spare-in l31) (spare-in l44) (spare-in l47) (spare-in l48) (spare-in l50) (spare-in l6) (vehicle-at l3))
    (:goal (vehicle-at l7))
)