(define (problem tireworld-056-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l6 l7 l8 l9 - location)
    (:init (road l0 l25) (road l1 l41) (road l1 l53) (road l10 l40) (road l10 l54) (road l11 l22) (road l11 l30) (road l12 l24) (road l12 l5) (road l13 l21) (road l13 l6) (road l14 l48) (road l14 l49) (road l15 l31) (road l15 l44) (road l16 l3) (road l16 l41) (road l17 l24) (road l17 l9) (road l18 l27) (road l18 l32) (road l19 l4) (road l19 l52) (road l2 l3) (road l2 l43) (road l20 l30) (road l20 l42) (road l21 l13) (road l21 l7) (road l22 l11) (road l22 l45) (road l22 l8) (road l23 l34) (road l23 l50) (road l24 l12) (road l24 l17) (road l25 l0) (road l25 l26) (road l26 l25) (road l26 l43) (road l27 l18) (road l27 l48) (road l28 l38) (road l28 l51) (road l29 l37) (road l29 l5) (road l29 l50) (road l3 l16) (road l3 l2) (road l30 l11) (road l30 l20) (road l31 l15) (road l31 l46) (road l32 l18) (road l32 l35) (road l33 l52) (road l33 l9) (road l34 l23) (road l35 l32) (road l35 l37) (road l35 l6) (road l36 l42) (road l36 l44) (road l37 l29) (road l37 l35) (road l38 l28) (road l38 l48) (road l39 l4) (road l39 l40) (road l4 l19) (road l4 l39) (road l40 l10) (road l40 l39) (road l41 l1) (road l41 l16) (road l42 l20) (road l42 l36) (road l43 l2) (road l43 l26) (road l44 l15) (road l44 l36) (road l45 l22) (road l45 l53) (road l46 l31) (road l46 l47) (road l47 l46) (road l47 l55) (road l48 l14) (road l48 l27) (road l48 l38) (road l49 l14) (road l49 l55) (road l5 l12) (road l5 l29) (road l50 l23) (road l50 l29) (road l51 l28) (road l51 l8) (road l52 l19) (road l52 l33) (road l53 l1) (road l53 l45) (road l54 l10) (road l54 l7) (road l55 l47) (road l55 l49) (road l6 l13) (road l6 l35) (road l7 l21) (road l7 l54) (road l8 l22) (road l8 l51) (road l9 l17) (road l9 l33) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l35) (spare-in l36) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l0))
    (:goal (vehicle-at l34))
)