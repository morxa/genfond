(define (problem tireworld-056-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l6 l7 l8 l9 - location)
    (:init (road l0 l46) (road l0 l48) (road l1 l24) (road l1 l43) (road l10 l2) (road l10 l6) (road l11 l22) (road l11 l33) (road l12 l16) (road l12 l31) (road l13 l23) (road l13 l39) (road l14 l34) (road l14 l54) (road l15 l33) (road l15 l44) (road l16 l12) (road l16 l45) (road l17 l34) (road l17 l7) (road l18 l27) (road l18 l45) (road l19 l3) (road l19 l32) (road l2 l10) (road l2 l9) (road l20 l31) (road l20 l43) (road l20 l45) (road l21 l35) (road l21 l44) (road l22 l11) (road l22 l6) (road l23 l13) (road l23 l8) (road l24 l1) (road l24 l28) (road l25 l29) (road l25 l38) (road l26 l30) (road l26 l32) (road l27 l18) (road l27 l38) (road l28 l24) (road l28 l49) (road l28 l54) (road l28 l55) (road l29 l25) (road l29 l41) (road l3 l19) (road l3 l9) (road l30 l26) (road l30 l7) (road l31 l12) (road l31 l20) (road l32 l19) (road l32 l26) (road l33 l11) (road l33 l15) (road l34 l14) (road l34 l17) (road l35 l21) (road l35 l37) (road l36 l47) (road l36 l55) (road l37 l35) (road l37 l42) (road l38 l25) (road l38 l27) (road l39 l13) (road l39 l42) (road l4 l51) (road l4 l53) (road l40 l45) (road l40 l53) (road l41 l29) (road l41 l8) (road l42 l37) (road l42 l39) (road l43 l1) (road l43 l20) (road l44 l15) (road l44 l21) (road l45 l16) (road l45 l18) (road l45 l20) (road l45 l40) (road l46 l0) (road l46 l51) (road l47 l36) (road l47 l5) (road l48 l0) (road l49 l28) (road l49 l50) (road l5 l47) (road l5 l52) (road l50 l49) (road l50 l52) (road l51 l4) (road l51 l46) (road l52 l5) (road l52 l50) (road l53 l4) (road l53 l40) (road l54 l14) (road l54 l28) (road l55 l28) (road l55 l36) (road l6 l10) (road l6 l22) (road l7 l17) (road l7 l30) (road l8 l23) (road l8 l41) (road l9 l2) (road l9 l3) (spare-in l0) (spare-in l1) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l20) (spare-in l21) (spare-in l24) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (vehicle-at l47))
    (:goal (vehicle-at l48))
)