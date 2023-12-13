(define (problem tireworld-080-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l9 - location)
    (:init (road l0 l6) (road l0 l78) (road l1 l16) (road l1 l66) (road l10 l26) (road l10 l39) (road l11 l35) (road l11 l42) (road l12 l51) (road l12 l76) (road l13 l6) (road l13 l8) (road l14 l29) (road l14 l67) (road l15 l39) (road l15 l9) (road l16 l1) (road l16 l2) (road l16 l70) (road l17 l37) (road l17 l50) (road l17 l69) (road l18 l34) (road l18 l56) (road l19 l59) (road l2 l16) (road l2 l32) (road l20 l3) (road l20 l58) (road l21 l22) (road l21 l24) (road l22 l21) (road l22 l67) (road l23 l40) (road l23 l45) (road l24 l21) (road l24 l32) (road l25 l26) (road l25 l33) (road l26 l10) (road l26 l25) (road l26 l68) (road l27 l55) (road l27 l7) (road l28 l40) (road l28 l64) (road l29 l14) (road l29 l54) (road l29 l61) (road l3 l20) (road l3 l54) (road l30 l53) (road l30 l65) (road l31 l36) (road l31 l46) (road l32 l2) (road l32 l24) (road l32 l48) (road l33 l25) (road l34 l18) (road l34 l62) (road l35 l11) (road l35 l44) (road l36 l31) (road l36 l43) (road l37 l17) (road l37 l63) (road l38 l59) (road l38 l75) (road l39 l10) (road l39 l15) (road l4 l50) (road l4 l70) (road l4 l8) (road l40 l23) (road l40 l28) (road l40 l42) (road l41 l44) (road l41 l69) (road l42 l11) (road l42 l40) (road l43 l36) (road l43 l48) (road l44 l35) (road l44 l41) (road l45 l23) (road l45 l5) (road l46 l31) (road l46 l51) (road l47 l61) (road l47 l77) (road l48 l32) (road l48 l43) (road l49 l7) (road l49 l78) (road l5 l45) (road l5 l72) (road l50 l17) (road l50 l4) (road l51 l12) (road l51 l46) (road l52 l57) (road l52 l66) (road l53 l30) (road l53 l73) (road l54 l29) (road l54 l3) (road l55 l27) (road l55 l71) (road l56 l18) (road l56 l79) (road l57 l52) (road l57 l60) (road l58 l20) (road l58 l71) (road l59 l19) (road l59 l38) (road l59 l77) (road l6 l0) (road l6 l13) (road l60 l57) (road l60 l68) (road l61 l29) (road l61 l47) (road l62 l34) (road l62 l76) (road l63 l37) (road l63 l79) (road l64 l28) (road l64 l74) (road l65 l30) (road l65 l75) (road l66 l1) (road l66 l52) (road l67 l14) (road l67 l22) (road l68 l26) (road l68 l60) (road l69 l17) (road l69 l41) (road l7 l27) (road l7 l49) (road l70 l16) (road l70 l4) (road l71 l55) (road l71 l58) (road l72 l5) (road l72 l9) (road l73 l53) (road l73 l74) (road l74 l64) (road l74 l73) (road l75 l38) (road l75 l65) (road l76 l12) (road l76 l62) (road l77 l47) (road l77 l59) (road l78 l0) (road l78 l49) (road l79 l56) (road l79 l63) (road l8 l13) (road l8 l4) (road l9 l15) (road l9 l72) (spare-in l0) (spare-in l1) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l20) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l37) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (vehicle-at l19))
    (:goal (vehicle-at l33))
)