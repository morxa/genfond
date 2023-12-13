(define (problem tireworld-056-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l6 l7 l8 l9 - location)
    (:init (road l0 l11) (road l0 l52) (road l1 l18) (road l1 l50) (road l10 l13) (road l10 l33) (road l11 l0) (road l11 l13) (road l11 l45) (road l12 l19) (road l12 l23) (road l13 l10) (road l13 l11) (road l13 l22) (road l14 l15) (road l14 l8) (road l15 l14) (road l15 l23) (road l16 l21) (road l16 l37) (road l17 l41) (road l17 l9) (road l18 l1) (road l18 l43) (road l19 l12) (road l19 l29) (road l19 l30) (road l19 l33) (road l19 l43) (road l2 l49) (road l2 l50) (road l20 l41) (road l20 l47) (road l21 l16) (road l21 l6) (road l22 l13) (road l22 l42) (road l23 l12) (road l23 l15) (road l24 l32) (road l24 l34) (road l25 l3) (road l25 l35) (road l26 l36) (road l26 l45) (road l27 l43) (road l27 l55) (road l28 l55) (road l29 l19) (road l29 l3) (road l3 l25) (road l3 l29) (road l30 l19) (road l30 l55) (road l31 l34) (road l31 l8) (road l32 l24) (road l32 l52) (road l33 l10) (road l33 l19) (road l34 l24) (road l34 l31) (road l35 l25) (road l35 l53) (road l35 l6) (road l36 l26) (road l36 l4) (road l37 l16) (road l37 l44) (road l38 l46) (road l38 l49) (road l39 l54) (road l39 l7) (road l4 l36) (road l4 l46) (road l40 l51) (road l40 l7) (road l41 l17) (road l41 l20) (road l42 l22) (road l42 l55) (road l43 l18) (road l43 l19) (road l43 l27) (road l44 l37) (road l44 l9) (road l45 l11) (road l45 l26) (road l45 l53) (road l46 l38) (road l46 l4) (road l47 l20) (road l47 l5) (road l48 l51) (road l48 l6) (road l49 l2) (road l49 l38) (road l5 l47) (road l5 l50) (road l50 l1) (road l50 l2) (road l50 l5) (road l51 l40) (road l51 l48) (road l52 l0) (road l52 l32) (road l53 l35) (road l53 l45) (road l54 l39) (road l55 l27) (road l55 l28) (road l55 l30) (road l55 l42) (road l6 l21) (road l6 l35) (road l6 l48) (road l7 l39) (road l7 l40) (road l8 l14) (road l8 l31) (road l9 l17) (road l9 l44) (spare-in l11) (spare-in l13) (spare-in l15) (spare-in l17) (spare-in l19) (spare-in l22) (spare-in l24) (spare-in l30) (spare-in l35) (spare-in l39) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l45) (spare-in l48) (spare-in l51) (spare-in l53) (spare-in l55) (spare-in l6) (spare-in l7) (vehicle-at l54))
    (:goal (vehicle-at l28))
)