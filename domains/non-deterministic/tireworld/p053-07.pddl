(define (problem tireworld-053-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l6 l7 l8 l9 - location)
    (:init (road l0 l22) (road l0 l25) (road l1 l15) (road l1 l40) (road l10 l14) (road l10 l38) (road l11 l16) (road l11 l45) (road l12 l3) (road l12 l39) (road l13 l26) (road l13 l7) (road l14 l10) (road l14 l50) (road l15 l1) (road l15 l36) (road l16 l11) (road l16 l19) (road l17 l30) (road l17 l50) (road l18 l27) (road l18 l5) (road l19 l16) (road l19 l43) (road l2 l38) (road l2 l4) (road l20 l32) (road l20 l9) (road l21 l23) (road l21 l39) (road l22 l0) (road l22 l49) (road l23 l21) (road l23 l43) (road l23 l5) (road l24 l41) (road l24 l42) (road l25 l0) (road l25 l35) (road l26 l13) (road l26 l46) (road l27 l18) (road l27 l34) (road l27 l39) (road l28 l45) (road l28 l47) (road l29 l33) (road l29 l37) (road l3 l12) (road l3 l37) (road l30 l17) (road l30 l44) (road l31 l46) (road l31 l51) (road l32 l20) (road l32 l6) (road l33 l29) (road l33 l4) (road l33 l52) (road l34 l27) (road l35 l25) (road l35 l42) (road l36 l15) (road l36 l9) (road l37 l29) (road l37 l3) (road l38 l10) (road l38 l2) (road l39 l12) (road l39 l21) (road l39 l27) (road l4 l2) (road l4 l33) (road l40 l1) (road l40 l51) (road l41 l24) (road l41 l8) (road l42 l24) (road l42 l35) (road l43 l19) (road l43 l23) (road l44 l30) (road l44 l5) (road l45 l11) (road l45 l28) (road l46 l26) (road l46 l31) (road l47 l28) (road l47 l7) (road l48 l49) (road l48 l6) (road l49 l22) (road l49 l48) (road l5 l18) (road l5 l23) (road l5 l44) (road l50 l14) (road l50 l17) (road l51 l31) (road l51 l40) (road l52 l33) (road l52 l8) (road l6 l32) (road l6 l48) (road l7 l13) (road l7 l47) (road l8 l41) (road l8 l52) (road l9 l20) (road l9 l36) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l6) (spare-in l7) (spare-in l8) (spare-in l9) (vehicle-at l34))
    (:goal (vehicle-at l5))
)