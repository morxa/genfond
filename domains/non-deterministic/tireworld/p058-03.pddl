(define (problem tireworld-058-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l6 l7 l8 l9 - location)
    (:init (road l0 l10) (road l0 l31) (road l1 l16) (road l1 l56) (road l10 l0) (road l10 l38) (road l11 l50) (road l11 l57) (road l12 l15) (road l13 l17) (road l13 l53) (road l14 l18) (road l14 l22) (road l15 l12) (road l15 l23) (road l16 l1) (road l16 l33) (road l17 l13) (road l17 l5) (road l18 l14) (road l18 l24) (road l18 l56) (road l19 l23) (road l19 l37) (road l2 l25) (road l2 l4) (road l20 l42) (road l20 l49) (road l21 l43) (road l21 l46) (road l22 l14) (road l22 l39) (road l23 l15) (road l23 l19) (road l24 l18) (road l24 l51) (road l25 l2) (road l25 l8) (road l26 l50) (road l26 l7) (road l27 l29) (road l27 l3) (road l28 l32) (road l28 l5) (road l29 l27) (road l29 l34) (road l3 l27) (road l3 l39) (road l30 l46) (road l30 l48) (road l31 l0) (road l31 l54) (road l32 l28) (road l32 l44) (road l33 l16) (road l33 l37) (road l34 l29) (road l34 l36) (road l35 l38) (road l35 l45) (road l36 l34) (road l36 l8) (road l37 l19) (road l37 l33) (road l38 l10) (road l38 l35) (road l38 l43) (road l39 l22) (road l39 l3) (road l4 l2) (road l4 l5) (road l40 l48) (road l40 l5) (road l41 l54) (road l41 l6) (road l42 l20) (road l42 l6) (road l43 l21) (road l43 l38) (road l44 l32) (road l44 l45) (road l45 l35) (road l45 l44) (road l45 l53) (road l46 l21) (road l46 l30) (road l47 l51) (road l47 l52) (road l48 l30) (road l48 l40) (road l49 l20) (road l49 l55) (road l49 l9) (road l5 l17) (road l5 l28) (road l5 l4) (road l5 l40) (road l50 l11) (road l50 l26) (road l51 l24) (road l51 l47) (road l52 l47) (road l52 l55) (road l53 l13) (road l53 l45) (road l54 l31) (road l54 l41) (road l54 l57) (road l55 l49) (road l55 l52) (road l56 l1) (road l56 l18) (road l57 l11) (road l57 l54) (road l6 l41) (road l6 l42) (road l7 l26) (road l7 l9) (road l8 l25) (road l8 l36) (road l9 l49) (road l9 l7) (spare-in l1) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l3) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l41) (spare-in l45) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l6) (vehicle-at l12))
    (:goal (vehicle-at l42))
)