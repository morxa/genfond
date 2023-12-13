(define (problem tireworld-055-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l6 l7 l8 l9 - location)
    (:init (road l0 l30) (road l0 l45) (road l1 l36) (road l1 l43) (road l10 l54) (road l10 l9) (road l11 l37) (road l11 l52) (road l12 l18) (road l12 l32) (road l13 l3) (road l13 l48) (road l14 l17) (road l14 l21) (road l14 l40) (road l15 l25) (road l15 l42) (road l15 l53) (road l16 l38) (road l16 l46) (road l16 l6) (road l17 l14) (road l17 l35) (road l17 l53) (road l18 l12) (road l18 l49) (road l18 l54) (road l19 l29) (road l19 l32) (road l2 l34) (road l20 l39) (road l20 l41) (road l21 l14) (road l21 l7) (road l22 l48) (road l22 l9) (road l23 l27) (road l23 l37) (road l24 l30) (road l24 l52) (road l25 l15) (road l25 l4) (road l26 l28) (road l26 l40) (road l27 l23) (road l27 l36) (road l28 l26) (road l28 l8) (road l29 l19) (road l29 l38) (road l3 l13) (road l3 l8) (road l30 l0) (road l30 l24) (road l31 l38) (road l31 l7) (road l32 l12) (road l32 l19) (road l33 l47) (road l33 l6) (road l34 l2) (road l34 l49) (road l35 l17) (road l35 l50) (road l36 l1) (road l36 l27) (road l37 l11) (road l37 l23) (road l37 l41) (road l38 l16) (road l38 l29) (road l38 l31) (road l39 l20) (road l39 l46) (road l4 l25) (road l4 l44) (road l40 l14) (road l40 l26) (road l41 l20) (road l41 l37) (road l42 l15) (road l42 l51) (road l43 l1) (road l43 l50) (road l44 l4) (road l44 l8) (road l45 l0) (road l45 l5) (road l46 l16) (road l46 l39) (road l47 l33) (road l48 l13) (road l48 l22) (road l49 l18) (road l49 l34) (road l5 l45) (road l5 l51) (road l50 l35) (road l50 l43) (road l51 l42) (road l51 l5) (road l52 l11) (road l52 l24) (road l53 l15) (road l53 l17) (road l54 l10) (road l54 l18) (road l6 l16) (road l6 l33) (road l7 l21) (road l7 l31) (road l8 l28) (road l8 l3) (road l8 l44) (road l9 l10) (road l9 l22) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l47))
    (:goal (vehicle-at l2))
)