(define (problem tireworld-058-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l6 l7 l8 l9 - location)
    (:init (road l0 l20) (road l0 l34) (road l1 l4) (road l1 l54) (road l10 l28) (road l10 l3) (road l11 l33) (road l11 l40) (road l12 l32) (road l12 l55) (road l13 l35) (road l13 l48) (road l14 l42) (road l14 l5) (road l14 l52) (road l15 l30) (road l15 l53) (road l15 l7) (road l16 l24) (road l16 l45) (road l17 l2) (road l17 l47) (road l18 l43) (road l18 l46) (road l19 l46) (road l19 l9) (road l2 l17) (road l2 l39) (road l20 l0) (road l20 l44) (road l21 l43) (road l21 l44) (road l22 l30) (road l22 l51) (road l23 l27) (road l23 l38) (road l23 l52) (road l23 l54) (road l23 l56) (road l24 l16) (road l24 l36) (road l24 l6) (road l25 l40) (road l25 l55) (road l26 l31) (road l26 l50) (road l27 l23) (road l27 l28) (road l28 l10) (road l28 l27) (road l29 l32) (road l29 l34) (road l3 l10) (road l3 l57) (road l30 l15) (road l30 l22) (road l31 l26) (road l31 l49) (road l32 l12) (road l32 l29) (road l33 l11) (road l33 l8) (road l34 l0) (road l34 l29) (road l35 l13) (road l35 l53) (road l36 l24) (road l37 l38) (road l37 l9) (road l38 l23) (road l38 l37) (road l39 l2) (road l39 l57) (road l4 l1) (road l4 l48) (road l40 l11) (road l40 l25) (road l41 l48) (road l41 l49) (road l41 l6) (road l42 l14) (road l42 l50) (road l43 l18) (road l43 l21) (road l44 l20) (road l44 l21) (road l45 l16) (road l45 l56) (road l46 l18) (road l46 l19) (road l47 l17) (road l47 l51) (road l48 l13) (road l48 l4) (road l48 l41) (road l49 l31) (road l49 l41) (road l5 l14) (road l5 l8) (road l50 l26) (road l50 l42) (road l51 l22) (road l51 l47) (road l52 l14) (road l52 l23) (road l53 l15) (road l53 l35) (road l54 l1) (road l54 l23) (road l55 l12) (road l55 l25) (road l56 l23) (road l56 l45) (road l57 l3) (road l57 l39) (road l6 l24) (road l6 l41) (road l7 l15) (road l8 l33) (road l8 l5) (road l9 l19) (road l9 l37) (spare-in l10) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l35) (spare-in l37) (spare-in l39) (spare-in l45) (spare-in l47) (spare-in l48) (spare-in l5) (spare-in l51) (spare-in l54) (spare-in l56) (spare-in l57) (vehicle-at l36))
    (:goal (vehicle-at l7))
)