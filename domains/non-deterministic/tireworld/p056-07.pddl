(define (problem tireworld-056-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l6 l7 l8 l9 - location)
    (:init (road l0 l12) (road l0 l51) (road l1 l31) (road l1 l35) (road l10 l15) (road l10 l25) (road l10 l27) (road l11 l33) (road l11 l8) (road l12 l0) (road l12 l4) (road l12 l54) (road l13 l38) (road l13 l41) (road l14 l29) (road l14 l47) (road l15 l10) (road l15 l37) (road l16 l17) (road l16 l19) (road l16 l2) (road l16 l21) (road l16 l50) (road l16 l7) (road l17 l16) (road l17 l19) (road l17 l50) (road l18 l37) (road l18 l46) (road l19 l16) (road l19 l17) (road l19 l27) (road l2 l16) (road l2 l5) (road l20 l36) (road l20 l46) (road l21 l16) (road l21 l42) (road l22 l29) (road l22 l39) (road l23 l3) (road l23 l33) (road l24 l4) (road l24 l55) (road l25 l10) (road l25 l54) (road l26 l49) (road l26 l52) (road l27 l10) (road l27 l19) (road l28 l55) (road l28 l9) (road l29 l14) (road l29 l22) (road l3 l23) (road l3 l30) (road l30 l3) (road l30 l45) (road l31 l1) (road l31 l39) (road l32 l36) (road l32 l48) (road l33 l11) (road l33 l23) (road l34 l41) (road l34 l45) (road l35 l1) (road l35 l43) (road l36 l20) (road l36 l32) (road l37 l15) (road l37 l18) (road l38 l13) (road l38 l42) (road l39 l22) (road l39 l31) (road l4 l12) (road l4 l24) (road l40 l44) (road l40 l49) (road l40 l7) (road l41 l13) (road l41 l34) (road l42 l21) (road l42 l38) (road l43 l35) (road l43 l49) (road l44 l40) (road l44 l51) (road l45 l30) (road l45 l34) (road l46 l18) (road l46 l20) (road l47 l14) (road l47 l5) (road l48 l32) (road l48 l8) (road l49 l26) (road l49 l40) (road l49 l43) (road l5 l2) (road l5 l47) (road l50 l16) (road l50 l17) (road l51 l0) (road l51 l44) (road l51 l53) (road l52 l26) (road l52 l9) (road l53 l51) (road l53 l6) (road l54 l12) (road l54 l25) (road l55 l24) (road l55 l28) (road l6 l53) (road l7 l16) (road l7 l40) (road l8 l11) (road l8 l48) (road l9 l28) (road l9 l52) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l25) (spare-in l26) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l35) (spare-in l36) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l42) (spare-in l43) (spare-in l45) (spare-in l47) (spare-in l48) (spare-in l5) (spare-in l51) (spare-in l53) (spare-in l54) (vehicle-at l27))
    (:goal (vehicle-at l6))
)