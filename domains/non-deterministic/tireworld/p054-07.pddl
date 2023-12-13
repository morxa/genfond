(define (problem tireworld-054-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l6 l7 l8 l9 - location)
    (:init (road l0 l11) (road l0 l30) (road l1 l37) (road l1 l43) (road l10 l27) (road l10 l50) (road l11 l0) (road l11 l2) (road l12 l26) (road l12 l33) (road l13 l40) (road l13 l41) (road l14 l19) (road l14 l52) (road l15 l23) (road l15 l51) (road l16 l27) (road l16 l28) (road l17 l5) (road l18 l22) (road l18 l39) (road l18 l46) (road l19 l14) (road l19 l34) (road l2 l11) (road l2 l51) (road l20 l3) (road l20 l38) (road l21 l35) (road l21 l9) (road l22 l18) (road l22 l30) (road l23 l15) (road l23 l24) (road l24 l23) (road l24 l32) (road l25 l41) (road l25 l8) (road l26 l12) (road l26 l44) (road l27 l10) (road l27 l16) (road l28 l16) (road l28 l38) (road l28 l52) (road l29 l49) (road l29 l7) (road l3 l20) (road l3 l42) (road l30 l0) (road l30 l22) (road l30 l48) (road l31 l35) (road l31 l36) (road l31 l42) (road l32 l24) (road l32 l4) (road l33 l12) (road l33 l48) (road l34 l19) (road l34 l49) (road l35 l21) (road l35 l31) (road l35 l39) (road l36 l31) (road l36 l6) (road l37 l1) (road l37 l45) (road l38 l20) (road l38 l28) (road l39 l18) (road l39 l35) (road l39 l46) (road l39 l50) (road l4 l32) (road l4 l40) (road l40 l13) (road l40 l4) (road l41 l13) (road l41 l25) (road l42 l3) (road l42 l31) (road l43 l1) (road l43 l6) (road l44 l26) (road l44 l53) (road l45 l37) (road l45 l47) (road l46 l18) (road l46 l39) (road l46 l9) (road l47 l45) (road l47 l5) (road l48 l30) (road l48 l33) (road l49 l29) (road l49 l34) (road l5 l17) (road l5 l47) (road l50 l10) (road l50 l39) (road l51 l15) (road l51 l2) (road l52 l14) (road l52 l28) (road l53 l44) (road l6 l36) (road l6 l43) (road l7 l29) (road l7 l8) (road l8 l25) (road l8 l7) (road l9 l21) (road l9 l46) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l16) (spare-in l18) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l33) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l5) (spare-in l50) (spare-in l6) (vehicle-at l17))
    (:goal (vehicle-at l53))
)