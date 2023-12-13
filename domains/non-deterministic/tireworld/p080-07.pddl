(define (problem tireworld-080-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l9 - location)
    (:init (road l0 l6) (road l0 l63) (road l1 l31) (road l1 l59) (road l10 l57) (road l10 l79) (road l11 l32) (road l11 l55) (road l12 l42) (road l12 l49) (road l13 l3) (road l13 l73) (road l14 l33) (road l14 l62) (road l15 l16) (road l15 l7) (road l16 l15) (road l16 l43) (road l16 l44) (road l17 l2) (road l17 l41) (road l18 l26) (road l18 l69) (road l19 l29) (road l19 l57) (road l2 l17) (road l2 l58) (road l20 l61) (road l20 l65) (road l21 l24) (road l21 l7) (road l22 l35) (road l22 l49) (road l23 l36) (road l23 l53) (road l24 l21) (road l24 l27) (road l25 l34) (road l25 l45) (road l26 l18) (road l26 l39) (road l26 l61) (road l27 l24) (road l27 l51) (road l28 l54) (road l28 l64) (road l29 l19) (road l29 l38) (road l3 l13) (road l3 l53) (road l3 l63) (road l30 l60) (road l30 l68) (road l31 l1) (road l31 l78) (road l32 l11) (road l32 l67) (road l33 l14) (road l33 l66) (road l34 l25) (road l34 l73) (road l35 l22) (road l35 l51) (road l36 l23) (road l36 l70) (road l37 l4) (road l37 l77) (road l38 l29) (road l38 l55) (road l39 l26) (road l39 l75) (road l4 l37) (road l4 l8) (road l40 l50) (road l40 l64) (road l41 l17) (road l41 l52) (road l42 l12) (road l42 l76) (road l43 l16) (road l43 l60) (road l43 l62) (road l44 l16) (road l44 l52) (road l45 l25) (road l45 l66) (road l46 l48) (road l47 l5) (road l47 l75) (road l48 l46) (road l48 l72) (road l49 l12) (road l49 l22) (road l5 l47) (road l5 l56) (road l50 l40) (road l50 l73) (road l51 l27) (road l51 l35) (road l52 l41) (road l52 l44) (road l52 l9) (road l53 l23) (road l53 l3) (road l54 l28) (road l55 l11) (road l55 l38) (road l56 l5) (road l56 l71) (road l57 l10) (road l57 l19) (road l58 l2) (road l58 l70) (road l59 l1) (road l59 l67) (road l6 l0) (road l6 l79) (road l60 l30) (road l60 l43) (road l61 l20) (road l61 l26) (road l62 l14) (road l62 l43) (road l63 l0) (road l63 l3) (road l64 l28) (road l64 l40) (road l65 l20) (road l65 l78) (road l66 l33) (road l66 l45) (road l67 l32) (road l67 l59) (road l68 l30) (road l68 l74) (road l69 l18) (road l69 l76) (road l7 l15) (road l7 l21) (road l70 l36) (road l70 l58) (road l71 l56) (road l71 l77) (road l72 l48) (road l72 l8) (road l73 l13) (road l73 l34) (road l73 l50) (road l74 l68) (road l74 l9) (road l75 l39) (road l75 l47) (road l76 l42) (road l76 l69) (road l77 l37) (road l77 l71) (road l78 l31) (road l78 l65) (road l79 l10) (road l79 l6) (road l8 l4) (road l8 l72) (road l9 l52) (road l9 l74) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l13) (spare-in l16) (spare-in l19) (spare-in l20) (spare-in l22) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l47) (spare-in l48) (spare-in l5) (spare-in l50) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l7) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l75) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (vehicle-at l46))
    (:goal (vehicle-at l54))
)